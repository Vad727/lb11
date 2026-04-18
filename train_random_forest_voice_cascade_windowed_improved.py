from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# НАСТРОЙКИ ПРОЕКТА
# Пути и названия папок оставлены в совместимом виде.
# ============================================================

TRAIN_CLASS_FOLDERS: Dict[str, str] = {
    "adult_male": r"/home/test/Рабочий стол/Классификатор пола/данные/dataset/adult_male",
    "adult_female": r"/home/test/Рабочий стол/Классификатор пола/данные/dataset/adult_female",
    "child_boy": r"/home/test/Рабочий стол/Классификатор пола/данные/dataset/child_boy",
    "child_girl": r"/home/test/Рабочий стол/Классификатор пола/данные/dataset/child_girl",
}

METADATA_CSV: str = r"/home/test/Рабочий стол/Классификатор пола/данные/metadata.csv"
OUTPUT_DIR: str = r"/home/test/Рабочий стол/Классификатор пола/random_forest_window/random_forest_window_results"

FOUR_CLASS_LABELS: List[str] = ["adult_male", "adult_female", "child_boy", "child_girl"]


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    top_db: float = 25.0
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    min_duration_sec: float = 0.30
    target_rms: float = 0.1
    yin_fmin: float = 50.0
    yin_fmax: float = 800.0
    n_mfcc: int = 13
    formant_order: int = 12
    max_formant_frames: int = 200
    window_sec: float = 1.0
    window_hop_sec: float = 0.5
    min_window_voiced_ratio: float = 0.30
    fallback_window_voiced_ratio: float = 0.15
    max_windows_per_file_train: int = 5
    max_windows_per_file_infer: int = 5
    probability_aggregation: str = "mean"  # mean | median


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    criterion: str = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    class_weight: str = "balanced"
    n_jobs: int = -1


class PathUtils:
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_audio_file(path: Path) -> bool:
        return path.suffix.lower() in PathUtils.AUDIO_EXTS


class FeatureExtractor:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    @property
    def feature_names(self) -> List[str]:
        names = [
            "f0_median", "f0_iqr", "f0_p10", "f0_p90", "voiced_ratio",
            "f1_median", "f2_median", "f3_median", "f2_minus_f1_median",
            "f3_minus_f2_median", "formant_dispersion",
        ]
        for idx in range(1, self.config.n_mfcc + 1):
            names.extend([f"mfcc_{idx:02d}_mean", f"mfcc_{idx:02d}_std"])
        names.extend([
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_bandwidth_mean", "spectral_bandwidth_std",
            "spectral_rolloff_mean", "spectral_rolloff_std",
            "zcr_mean", "zcr_std",
            "rms_energy_mean", "rms_energy_std",
            "spectral_energy_mean", "spectral_energy_std",
            "log_spectral_energy_mean", "log_spectral_energy_std",
        ])
        return names

    @staticmethod
    def _robust_mean(values: np.ndarray, default: float = 0.0) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float(default)

    @staticmethod
    def _robust_std(values: np.ndarray, default: float = 0.0) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.std(values)) if values.size else float(default)

    @staticmethod
    def _robust_median(values: np.ndarray, default: float = 0.0) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float(default)

    @staticmethod
    def _robust_iqr(values: np.ndarray, default: float = 0.0) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else float(default)

    @staticmethod
    def _robust_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.percentile(values, q)) if values.size else float(default)

    @staticmethod
    def _pad_signal_to_length(y: np.ndarray, min_length: int) -> np.ndarray:
        if len(y) >= min_length:
            return y
        return np.pad(y, (0, min_length - len(y)), mode="constant")

    def _normalize_rms(self, y: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(y))) if len(y) else 0.0)
        if rms <= 1e-8:
            return y
        return y * (self.config.target_rms / rms)

    @staticmethod
    def _frame_signal_for_lpc(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        pad = frame_length // 2
        y_pad = np.pad(y, (pad, pad), mode="reflect")
        return librosa.util.frame(y_pad, frame_length=frame_length, hop_length=hop_length)

    def _estimate_formants_from_frame(self, frame: np.ndarray, sr: int) -> Tuple[float, float, float]:
        frame = np.asarray(frame, dtype=float)
        if frame.size < max(self.config.formant_order + 2, 32):
            return np.nan, np.nan, np.nan
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-6:
            return np.nan, np.nan, np.nan

        preemphasized = lfilter([1.0, -0.97], [1.0], frame)
        windowed = preemphasized * np.hamming(len(preemphasized))
        try:
            a = librosa.lpc(windowed, order=self.config.formant_order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            if roots.size == 0:
                return np.nan, np.nan, np.nan

            angs = np.arctan2(np.imag(roots), np.real(roots))
            freqs = angs * (sr / (2 * np.pi))
            bandwidths = -0.5 * (sr / np.pi) * np.log(np.maximum(np.abs(roots), 1e-12))
            valid = (freqs > 90) & (freqs < 5000) & (bandwidths < 700)
            freqs = np.sort(freqs[valid])
            if freqs.size < 3:
                return np.nan, np.nan, np.nan
            return float(freqs[0]), float(freqs[1]), float(freqs[2])
        except Exception:
            return np.nan, np.nan, np.nan

    def _estimate_formants(
        self,
        y: np.ndarray,
        sr: int,
        voiced_flags: np.ndarray,
        frame_length: int,
        hop_length: int,
    ) -> Dict[str, float]:
        frames = self._frame_signal_for_lpc(y, frame_length=frame_length, hop_length=hop_length)
        n_frames = min(frames.shape[1], len(voiced_flags))
        frames = frames[:, :n_frames]
        voiced_flags = np.asarray(voiced_flags[:n_frames], dtype=bool)

        empty = {
            "f1_median": 0.0,
            "f2_median": 0.0,
            "f3_median": 0.0,
            "f2_minus_f1_median": 0.0,
            "f3_minus_f2_median": 0.0,
            "formant_dispersion": 0.0,
        }
        if n_frames == 0:
            return empty

        voiced_indices = np.where(voiced_flags)[0]
        if voiced_indices.size == 0:
            return empty

        if voiced_indices.size > self.config.max_formant_frames:
            select_idx = np.linspace(0, voiced_indices.size - 1, self.config.max_formant_frames).astype(int)
            voiced_indices = voiced_indices[select_idx]

        f1_vals: List[float] = []
        f2_vals: List[float] = []
        f3_vals: List[float] = []
        for idx in voiced_indices:
            f1, f2, f3 = self._estimate_formants_from_frame(frames[:, idx], sr=sr)
            if np.isfinite(f1) and np.isfinite(f2) and np.isfinite(f3):
                f1_vals.append(f1)
                f2_vals.append(f2)
                f3_vals.append(f3)

        if not f1_vals:
            return empty

        f1_vals = np.asarray(f1_vals)
        f2_vals = np.asarray(f2_vals)
        f3_vals = np.asarray(f3_vals)
        diff_21 = f2_vals - f1_vals
        diff_32 = f3_vals - f2_vals
        return {
            "f1_median": self._robust_median(f1_vals),
            "f2_median": self._robust_median(f2_vals),
            "f3_median": self._robust_median(f3_vals),
            "f2_minus_f1_median": self._robust_median(diff_21),
            "f3_minus_f2_median": self._robust_median(diff_32),
            "formant_dispersion": self._robust_median(np.concatenate([diff_21, diff_32])),
        }

    def load_audio(self, audio_path: Path) -> np.ndarray:
        sr = self.config.sample_rate
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        y, _ = librosa.effects.trim(y, top_db=self.config.top_db)
        min_samples = int(self.config.min_duration_sec * sr)
        if len(y) < min_samples:
            y = self._pad_signal_to_length(y, min_samples)
        y = self._pad_signal_to_length(y, max(self.config.win_length, self.config.n_fft))
        y = self._normalize_rms(y)
        return y.astype(np.float32)

    def split_into_windows(self, y: np.ndarray) -> List[Tuple[int, float, np.ndarray]]:
        sr = self.config.sample_rate
        win_len = int(self.config.window_sec * sr)
        hop_len = int(self.config.window_hop_sec * sr)

        if len(y) <= win_len:
            y_pad = self._pad_signal_to_length(y, win_len)
            return [(0, 0.0, y_pad.astype(np.float32))]

        windows: List[Tuple[int, float, np.ndarray]] = []
        start = 0
        window_idx = 0
        while start + win_len <= len(y):
            window = y[start:start + win_len].astype(np.float32)
            windows.append((window_idx, start / sr, window))
            window_idx += 1
            start += hop_len

        if not windows:
            y_pad = self._pad_signal_to_length(y, win_len)
            windows.append((0, 0.0, y_pad.astype(np.float32)))

        return windows

    def extract_from_signal(self, y: np.ndarray) -> Dict[str, float]:
        sr = self.config.sample_rate
        frame_length = self.config.win_length
        hop_length = self.config.hop_length
        n_fft = self.config.n_fft

        y = np.asarray(y, dtype=np.float32)
        y = self._pad_signal_to_length(y, max(frame_length, n_fft))

        f0 = librosa.yin(
            y,
            fmin=self.config.yin_fmin,
            fmax=self.config.yin_fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        f0 = np.asarray(f0, dtype=float)
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
        zcr_frames = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length, center=True)[0]

        n_common = min(len(f0), len(rms_frames), len(zcr_frames))
        f0 = f0[:n_common]
        rms_frames = rms_frames[:n_common]
        zcr_frames = zcr_frames[:n_common]

        nonzero_rms = rms_frames[rms_frames > 1e-8]
        rms_threshold = float(0.1 * np.median(nonzero_rms)) if nonzero_rms.size else 1e-5
        voiced_flag = (rms_frames > max(rms_threshold, 1e-5)) & np.isfinite(f0)
        voiced_f0 = f0[voiced_flag]

        formants = self._estimate_formants(y, sr, voiced_flag, frame_length=frame_length, hop_length=hop_length)

        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, center=True)
        magnitude = np.abs(stft)
        power = magnitude ** 2
        spectral_energy = np.sum(power, axis=0)
        log_spectral_energy = np.log(spectral_energy + 1e-10)

        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85)[0]
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=frame_length,
            center=True,
        )

        features: Dict[str, float] = {
            "f0_median": self._robust_median(voiced_f0),
            "f0_iqr": self._robust_iqr(voiced_f0),
            "f0_p10": self._robust_percentile(voiced_f0, 10),
            "f0_p90": self._robust_percentile(voiced_f0, 90),
            "voiced_ratio": float(np.mean(voiced_flag.astype(float))) if voiced_flag.size else 0.0,
            "spectral_centroid_mean": self._robust_mean(spectral_centroid),
            "spectral_centroid_std": self._robust_std(spectral_centroid),
            "spectral_bandwidth_mean": self._robust_mean(spectral_bandwidth),
            "spectral_bandwidth_std": self._robust_std(spectral_bandwidth),
            "spectral_rolloff_mean": self._robust_mean(spectral_rolloff),
            "spectral_rolloff_std": self._robust_std(spectral_rolloff),
            "zcr_mean": self._robust_mean(zcr_frames),
            "zcr_std": self._robust_std(zcr_frames),
            "rms_energy_mean": self._robust_mean(rms_frames),
            "rms_energy_std": self._robust_std(rms_frames),
            "spectral_energy_mean": self._robust_mean(spectral_energy),
            "spectral_energy_std": self._robust_std(spectral_energy),
            "log_spectral_energy_mean": self._robust_mean(log_spectral_energy),
            "log_spectral_energy_std": self._robust_std(log_spectral_energy),
        }
        features.update(formants)

        for idx in range(self.config.n_mfcc):
            coeff = mfcc[idx]
            features[f"mfcc_{idx + 1:02d}_mean"] = self._robust_mean(coeff)
            features[f"mfcc_{idx + 1:02d}_std"] = self._robust_std(coeff)

        for feature_name in self.feature_names:
            features.setdefault(feature_name, 0.0)
            features[feature_name] = float(features[feature_name])
        return features

    def extract_windows(self, audio_path: Path) -> List[Dict[str, float]]:
        y = self.load_audio(audio_path)
        rows: List[Dict[str, float]] = []
        for window_idx, start_sec, window in self.split_into_windows(y):
            feats = self.extract_from_signal(window)
            feats["window_idx"] = int(window_idx)
            feats["window_start_sec"] = float(start_sec)
            rows.append(feats)
        return rows


class MetadataDatasetBuilder:
    def __init__(self, metadata_csv: Path, class_folders: Dict[str, str]) -> None:
        self.metadata_csv = metadata_csv
        self.class_folders = {label: Path(path).resolve() for label, path in class_folders.items()}

    def _resolve_path(self, raw_path: str, label: str) -> Path:
        candidate = Path(str(raw_path).strip())
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()

        metadata_relative = (self.metadata_csv.parent / candidate).resolve()
        if metadata_relative.exists():
            return metadata_relative

        if label in self.class_folders:
            label_relative = (self.class_folders[label] / candidate).resolve()
            if label_relative.exists():
                return label_relative
            by_name = (self.class_folders[label] / candidate.name).resolve()
            if by_name.exists():
                return by_name

        return metadata_relative

    def build(self) -> pd.DataFrame:
        meta = pd.read_csv(self.metadata_csv)
        required_columns = {"path", "label"}
        if not required_columns.issubset(meta.columns):
            raise ValueError("В metadata.csv должны быть столбцы: path и label")

        rows: List[Dict[str, str]] = []
        missing_paths: List[str] = []
        for _, row in meta.iterrows():
            label = str(row["label"]).strip()
            raw_path = str(row["path"]).strip()
            resolved = self._resolve_path(raw_path, label)
            if not resolved.exists():
                missing_paths.append(f"{raw_path} -> {resolved}")
                continue
            rows.append({"path": str(resolved), "label": label})

        if not rows:
            raise RuntimeError("Не удалось собрать ни одного файла из metadata.csv")

        dataset = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
        if missing_paths:
            print("[WARN] Часть файлов из metadata.csv не найдена. Примеры:")
            for item in missing_paths[:10]:
                print("   ", item)
        return dataset


class RandomForestVoiceClassifier:
    def __init__(self, train_config: TrainConfig) -> None:
        self.train_config = train_config
        self.scaler = StandardScaler()
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.classes_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.model = RandomForestClassifier(
            n_estimators=self.train_config.n_estimators,
            criterion=self.train_config.criterion,
            max_depth=self.train_config.max_depth,
            min_samples_split=self.train_config.min_samples_split,
            min_samples_leaf=self.train_config.min_samples_leaf,
            max_features=self.train_config.max_features,
            bootstrap=self.train_config.bootstrap,
            class_weight=self.train_config.class_weight,
            random_state=self.train_config.random_state,
            n_jobs=self.train_config.n_jobs,
        )
        self.model.fit(X_scaled, y.astype(str).values)
        self.classes_ = list(self.model.classes_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель Random Forest еще не обучена")
        X_scaled = self.scaler.transform(X[self.feature_names].values)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель Random Forest еще не обучена")
        X_scaled = self.scaler.transform(X[self.feature_names].values)
        return self.model.predict_proba(X_scaled)

    def save_stage_bundle(self) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Нельзя сохранить необученную модель Random Forest")
        return {
            "scaler": self.scaler,
            "model": self.model,
            "feature_names": self.feature_names,
            "classes": self.classes_,
        }


class PlotUtils:
    @staticmethod
    def make_barplot(data: pd.DataFrame, classes: List[str], column: str, title: str, output_path: Path) -> None:
        class_means = []
        for class_name in classes:
            values = data.loc[data["label"] == class_name, column].dropna().values
            class_means.append(float(np.mean(values)) if len(values) else np.nan)

        plt.figure(figsize=(8, 5))
        x = np.arange(len(classes))
        plt.bar(x, class_means)
        plt.title(title)
        plt.xlabel("Класс")
        plt.ylabel("Среднее значение признака")
        plt.xticks(x, classes, rotation=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()

    @staticmethod
    def make_histogram(data: pd.DataFrame, classes: List[str], column: str, title: str, output_path: Path) -> None:
        plt.figure(figsize=(8, 5))
        for class_name in classes:
            values = data.loc[data["label"] == class_name, column].dropna().values
            if len(values) == 0:
                continue
            bins = min(30, max(10, int(np.sqrt(len(values)))))
            plt.hist(values, bins=bins, density=True, alpha=0.45, label=class_name)
        plt.title(title)
        plt.xlabel("Значение признака")
        plt.ylabel("Плотность")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()


class VoiceRandomForestTrainer:
    def __init__(
        self,
        metadata_csv: Path,
        class_folders: Dict[str, str],
        output_dir: Path,
        audio_config: Optional[AudioConfig] = None,
        train_config: Optional[TrainConfig] = None,
    ) -> None:
        self.metadata_csv = metadata_csv.resolve()
        self.class_folders = class_folders
        self.output_dir = output_dir.resolve()
        self.audio_config = audio_config or AudioConfig()
        self.train_config = train_config or TrainConfig()
        self.extractor = FeatureExtractor(self.audio_config)

        self.tables_dir = self.output_dir / "tables"
        self.plots_dir = self.output_dir / "plots"
        self.model_dir = self.output_dir / "model"

    def _map_age_group(self, label: str) -> str:
        if label in ["adult_female", "adult_male"]:
            return "adult"
        if label in ["child_boy", "child_girl"]:
            return "child"
        raise ValueError(f"Неизвестная метка класса: {label}")

    def _select_informative_windows(self, windows_df: pd.DataFrame, max_windows: int) -> pd.DataFrame:
        df = windows_df.copy().reset_index(drop=True)
        if df.empty:
            return df

        voiced_ratio = df["voiced_ratio"].fillna(0.0).astype(float)
        f0_valid = (df["f0_median"].fillna(0.0).astype(float) > 0.0).astype(float)
        rms_vals = df["rms_energy_mean"].fillna(0.0).astype(float)

        nonzero_rms = rms_vals[rms_vals > 1e-8]
        median_rms = float(nonzero_rms.median()) if len(nonzero_rms) else 1.0
        if median_rms <= 1e-8:
            median_rms = 1.0
        rms_rel = np.clip(rms_vals / median_rms, 0.0, 2.0)

        df["window_score"] = voiced_ratio * np.sqrt(rms_rel) * (0.7 + 0.3 * f0_valid)
        df["is_selected"] = 0

        strict_mask = (voiced_ratio >= self.audio_config.min_window_voiced_ratio) & (f0_valid > 0)
        relaxed_mask = voiced_ratio >= self.audio_config.fallback_window_voiced_ratio

        if strict_mask.any():
            candidates = df[strict_mask].copy()
        elif relaxed_mask.any():
            candidates = df[relaxed_mask].copy()
        else:
            candidates = df.copy()

        candidates = candidates.sort_values(
            by=["window_score", "voiced_ratio", "rms_energy_mean", "window_idx"],
            ascending=[False, False, False, True],
        )
        selected_idx = candidates.head(max_windows).index
        df.loc[selected_idx, "is_selected"] = 1

        selected = df.loc[df["is_selected"] == 1].copy()
        if selected.empty:
            selected = df.sort_values(by=["window_idx"]).head(1).copy()
            selected["is_selected"] = 1
        return selected.reset_index(drop=True)

    def _aggregate_probabilities(self, combined: pd.DataFrame) -> pd.Series:
        if combined.empty:
            raise RuntimeError("Нет окон для агрегации вероятностей")
        mode = str(self.audio_config.probability_aggregation).lower().strip()
        if mode == "median":
            return combined.median(axis=0)
        return combined.mean(axis=0)

    def _build_feature_table(self, dataset_df: pd.DataFrame, max_windows_per_file: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: List[Dict[str, object]] = []
        skipped: List[Dict[str, str]] = []
        total = len(dataset_df)

        for idx, row in dataset_df.iterrows():
            path = Path(row["path"])
            label = str(row["label"])
            try:
                window_rows = self.extractor.extract_windows(path)
                if not window_rows:
                    raise RuntimeError("Не удалось получить ни одного окна из файла")

                windows_df = pd.DataFrame(window_rows)
                windows_df = self._select_informative_windows(windows_df, max_windows=max_windows_per_file)
                windows_df["path"] = str(path)
                windows_df["label"] = label
                windows_df["age_group"] = self._map_age_group(label)
                windows_df["total_windows_in_file"] = int(len(window_rows))
                windows_df["selected_windows_in_file"] = int(len(windows_df))
                rows.extend(windows_df.to_dict(orient="records"))

                if (idx + 1) == 1 or (idx + 1) % 25 == 0 or (idx + 1) == total:
                    print(f"[INFO] Обработано файлов: {idx + 1}/{total}")
            except Exception as exc:
                skipped.append({"path": str(path), "label": label, "reason": str(exc)})
                print(f"[SKIP] {path}: {exc}")

        if not rows:
            raise RuntimeError("Не удалось извлечь признаки ни из одного файла")
        return pd.DataFrame(rows), pd.DataFrame(skipped)

    def _predict_file_from_windows(
        self,
        windows_df: pd.DataFrame,
        age_classifier: RandomForestVoiceClassifier,
        adult_classifier: RandomForestVoiceClassifier,
        child_classifier: RandomForestVoiceClassifier,
    ) -> Tuple[Dict[str, object], pd.DataFrame]:
        feature_cols = self.extractor.feature_names
        selected_windows = self._select_informative_windows(
            windows_df,
            max_windows=self.audio_config.max_windows_per_file_infer,
        )
        X = selected_windows[feature_cols].copy()

        age_probs = pd.DataFrame(
            age_classifier.predict_proba(X),
            columns=age_classifier.classes_,
            index=selected_windows.index,
        )
        adult_probs = pd.DataFrame(
            adult_classifier.predict_proba(X),
            columns=adult_classifier.classes_,
            index=selected_windows.index,
        )
        child_probs = pd.DataFrame(
            child_classifier.predict_proba(X),
            columns=child_classifier.classes_,
            index=selected_windows.index,
        )

        combined = pd.DataFrame(index=selected_windows.index)
        combined["adult_male"] = age_probs["adult"] * adult_probs["adult_male"]
        combined["adult_female"] = age_probs["adult"] * adult_probs["adult_female"]
        combined["child_boy"] = age_probs["child"] * child_probs["child_boy"]
        combined["child_girl"] = age_probs["child"] * child_probs["child_girl"]

        agg_probs = self._aggregate_probabilities(combined)
        predicted_label = str(agg_probs.idxmax())

        result: Dict[str, object] = {
            "path": str(selected_windows["path"].iloc[0]),
            "label": str(selected_windows["label"].iloc[0]),
            "predicted_label": predicted_label,
            "n_windows": int(len(selected_windows)),
            "proba_male": float(agg_probs["adult_male"]),
            "proba_female": float(agg_probs["adult_female"]),
            "proba_boy": float(agg_probs["child_boy"]),
            "proba_girl": float(agg_probs["child_girl"]),
            "stage1_adult_mean": float(age_probs["adult"].mean()),
            "stage1_child_mean": float(age_probs["child"].mean()),
            "stage2_male_given_adult_mean": float(adult_probs["adult_male"].mean()),
            "stage2_female_given_adult_mean": float(adult_probs["adult_female"].mean()),
            "stage3_boy_given_child_mean": float(child_probs["child_boy"].mean()),
            "stage3_girl_given_child_mean": float(child_probs["child_girl"].mean()),
        }

        window_predictions = selected_windows[["path", "label", "window_idx", "window_start_sec", "window_score", "is_selected"]].copy()
        window_predictions["window_predicted_label"] = combined.idxmax(axis=1).astype(str)
        window_predictions["proba_male"] = combined["adult_male"].values
        window_predictions["proba_female"] = combined["adult_female"].values
        window_predictions["proba_boy"] = combined["child_boy"].values
        window_predictions["proba_girl"] = combined["child_girl"].values
        window_predictions["stage1_adult"] = age_probs["adult"].values
        window_predictions["stage1_child"] = age_probs["child"].values
        return result, window_predictions

    def _save_feature_tables(
        self,
        all_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        skipped_df: pd.DataFrame,
    ) -> None:
        PathUtils.ensure_dir(self.tables_dir)

        all_to_save = all_df.copy()
        all_to_save["split"] = "all"
        train_to_save = train_df.copy()
        train_to_save["split"] = "train"
        test_to_save = test_df.copy()
        test_to_save["split"] = "test"

        all_to_save.to_csv(self.tables_dir / "all_features.csv", index=False, encoding="utf-8-sig")
        train_to_save.to_csv(self.tables_dir / "train_features.csv", index=False, encoding="utf-8-sig")
        test_to_save.to_csv(self.tables_dir / "test_features.csv", index=False, encoding="utf-8-sig")

        core_cols = [
            "path", "label", "age_group", "window_idx", "window_start_sec",
            "window_score", "is_selected", "total_windows_in_file", "selected_windows_in_file",
            *feature_cols,
        ]
        train_to_save[core_cols].to_csv(self.tables_dir / "train_selected_features.csv", index=False, encoding="utf-8-sig")
        test_to_save[core_cols].to_csv(self.tables_dir / "test_selected_features.csv", index=False, encoding="utf-8-sig")

        file_counts = all_df[["path", "label"]].drop_duplicates()["label"].value_counts()
        window_counts = all_df["label"].value_counts()
        class_distribution = pd.DataFrame({
            "label": sorted(set(window_counts.index).union(set(file_counts.index))),
        })
        class_distribution["file_count"] = class_distribution["label"].map(file_counts).fillna(0).astype(int)
        class_distribution["window_count"] = class_distribution["label"].map(window_counts).fillna(0).astype(int)
        class_distribution.to_csv(self.tables_dir / "class_distribution.csv", index=False, encoding="utf-8-sig")

        if not skipped_df.empty:
            skipped_df.to_csv(self.tables_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")

    def _save_plots(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> None:
        PathUtils.ensure_dir(self.plots_dir)
        classes = FOUR_CLASS_LABELS.copy()
        split_map = {"train": train_df, "test": test_df}

        for split_name, split_df in split_map.items():
            split_root = self.plots_dir / split_name
            bars_dir = split_root / "bars"
            hist_dir = split_root / "histograms"
            PathUtils.ensure_dir(bars_dir)
            PathUtils.ensure_dir(hist_dir)

            for column in feature_cols:
                if column not in split_df.columns:
                    continue
                bar_path = bars_dir / f"{column}_bar.png"
                hist_path = hist_dir / f"{column}_histogram.png"
                PlotUtils.make_barplot(
                    split_df,
                    classes,
                    column,
                    title=f"{split_name.upper()}: средние значения по выбранным окнам для {column}",
                    output_path=bar_path,
                )
                PlotUtils.make_histogram(
                    split_df,
                    classes,
                    column,
                    title=f"{split_name.upper()}: гистограмма по выбранным окнам для {column}",
                    output_path=hist_path,
                )

    def _save_model_artifacts(
        self,
        age_classifier: RandomForestVoiceClassifier,
        adult_classifier: RandomForestVoiceClassifier,
        child_classifier: RandomForestVoiceClassifier,
        feature_cols: List[str],
        file_predictions_df: pd.DataFrame,
        window_predictions_df: pd.DataFrame,
        metrics: Dict[str, float],
        report_text: str,
        report_df: pd.DataFrame,
        cm_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Path:
        PathUtils.ensure_dir(self.model_dir)
        model_path = self.model_dir / "voice_random_forest_model_bundle.joblib"

        bundle = {
            "audio_config": asdict(self.audio_config),
            "train_config": asdict(self.train_config),
            "feature_names": feature_cols,
            "classes": FOUR_CLASS_LABELS,
            "model_type": "random_forest_cascade_windowed",
            "cascade": {
                "age_group": age_classifier.save_stage_bundle(),
                "adult_gender": adult_classifier.save_stage_bundle(),
                "child_gender": child_classifier.save_stage_bundle(),
            },
        }
        joblib.dump(bundle, model_path)

        with open(self.model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        with open(self.model_dir / "audio_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.audio_config), f, ensure_ascii=False, indent=2)
        with open(self.model_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.train_config), f, ensure_ascii=False, indent=2)
        with open(self.model_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(self.model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        report_df.to_csv(self.model_dir / "classification_report.csv", encoding="utf-8-sig")
        cm_df.to_csv(self.model_dir / "confusion_matrix.csv", encoding="utf-8-sig")
        file_predictions_df.to_csv(self.model_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
        window_predictions_df.to_csv(self.model_dir / "test_window_predictions.csv", index=False, encoding="utf-8-sig")

        for stage_name, clf in {
            "stage1_age_group": age_classifier,
            "stage2_adult_gender": adult_classifier,
            "stage3_child_gender": child_classifier,
        }.items():
            scaler_params_df = pd.DataFrame({
                "feature": feature_cols,
                "mean": clf.scaler.mean_,
                "scale": clf.scaler.scale_,
            })
            scaler_params_df.to_csv(self.model_dir / f"scaler_parameters_{stage_name}.csv", index=False, encoding="utf-8-sig")

        model_info = {
            "model_type": "random_forest_cascade_windowed",
            "n_classes_final": 4,
            "train_files": int(train_df[["path", "label"]].drop_duplicates().shape[0]),
            "test_files": int(test_df[["path", "label"]].drop_duplicates().shape[0]),
            "train_windows": int(len(train_df)),
            "test_windows": int(len(test_df)),
            "selection": {
                "min_window_voiced_ratio": self.audio_config.min_window_voiced_ratio,
                "fallback_window_voiced_ratio": self.audio_config.fallback_window_voiced_ratio,
                "max_windows_per_file_train": self.audio_config.max_windows_per_file_train,
                "max_windows_per_file_infer": self.audio_config.max_windows_per_file_infer,
                "probability_aggregation": self.audio_config.probability_aggregation,
            },
            "stages": {
                "age_group": {
                    "classes": age_classifier.classes_,
                    "n_estimators": age_classifier.model.n_estimators if age_classifier.model is not None else None,
                },
                "adult_gender": {
                    "classes": adult_classifier.classes_,
                    "n_estimators": adult_classifier.model.n_estimators if adult_classifier.model is not None else None,
                },
                "child_gender": {
                    "classes": child_classifier.classes_,
                    "n_estimators": child_classifier.model.n_estimators if child_classifier.model is not None else None,
                },
            },
        }
        with open(self.model_dir / "random_forest_model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        if age_classifier.model is not None:
            pd.DataFrame({
                "feature": feature_cols,
                "importance": age_classifier.model.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(
                self.model_dir / "feature_importance_stage1_age_group.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if adult_classifier.model is not None:
            pd.DataFrame({
                "feature": feature_cols,
                "importance": adult_classifier.model.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(
                self.model_dir / "feature_importance_stage2_adult_gender.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if child_classifier.model is not None:
            pd.DataFrame({
                "feature": feature_cols,
                "importance": child_classifier.model.feature_importances_,
            }).sort_values("importance", ascending=False).to_csv(
                self.model_dir / "feature_importance_stage3_child_gender.csv",
                index=False,
                encoding="utf-8-sig",
            )

        return model_path

    def _write_root_summary(
        self,
        all_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metrics: Dict[str, float],
        report_text: str,
        cm_df: pd.DataFrame,
        skipped_df: pd.DataFrame,
    ) -> None:
        train_files = train_df[["path", "label"]].drop_duplicates()
        test_files = test_df[["path", "label"]].drop_duplicates()
        file_distribution = all_df[["path", "label"]].drop_duplicates()["label"].value_counts()
        window_distribution = all_df["label"].value_counts()

        summary = [
            f"Размер таблицы признаков по выбранным окнам: {all_df.shape}",
            f"Количество файлов: {all_df[['path', 'label']].drop_duplicates().shape[0]}",
            "Количество файлов по классам:",
            file_distribution.to_string(),
            "",
            "Количество выбранных окон по классам:",
            window_distribution.to_string(),
            "",
            f"Train files: {len(train_files)}",
            f"Test files: {len(test_files)}",
            f"Train windows: {len(train_df)}",
            f"Test windows: {len(test_df)}",
            "",
            f"Accuracy (file-level): {metrics['accuracy']:.4f}",
            f"Macro F1 (file-level): {metrics['macro_f1']:.4f}",
            f"Weighted F1 (file-level): {metrics['weighted_f1']:.4f}",
            "",
            "Classification report:",
            report_text,
            "",
            "Confusion matrix:",
            cm_df.to_string(),
        ]
        if not skipped_df.empty:
            summary.extend(["", "Есть пропущенные файлы. См. tables/skipped_files.csv"])
        with open(self.output_dir / "RUN_SUMMARY.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary))

    def run(self) -> None:
        PathUtils.ensure_dir(self.output_dir)
        dataset_builder = MetadataDatasetBuilder(self.metadata_csv, self.class_folders)
        dataset_df = dataset_builder.build()
        if dataset_df["label"].nunique() != 4:
            print(f"[WARN] В metadata.csv найдено {dataset_df['label'].nunique()} классов, ожидалось 4.")

        train_files_df, test_files_df = train_test_split(
            dataset_df,
            test_size=self.train_config.test_size,
            random_state=self.train_config.random_state,
            stratify=dataset_df["label"],
        )
        train_files_df = train_files_df.reset_index(drop=True)
        test_files_df = test_files_df.reset_index(drop=True)

        print("[INFO] Извлекаю признаки из train-файлов по выбранным окнам...")
        train_df, skipped_train = self._build_feature_table(
            train_files_df,
            max_windows_per_file=self.audio_config.max_windows_per_file_train,
        )
        print("[INFO] Извлекаю признаки из test-файлов по выбранным окнам...")
        test_df, skipped_test = self._build_feature_table(
            test_files_df,
            max_windows_per_file=self.audio_config.max_windows_per_file_infer,
        )

        all_df = pd.concat([train_df, test_df], ignore_index=True)
        skipped_df = pd.concat([skipped_train, skipped_test], ignore_index=True)
        feature_names = self.extractor.feature_names

        print("\nРаспределение по классам (выбранные окна):")
        print(all_df["label"].value_counts())

        age_classifier = RandomForestVoiceClassifier(self.train_config)
        adult_classifier = RandomForestVoiceClassifier(self.train_config)
        child_classifier = RandomForestVoiceClassifier(self.train_config)

        age_classifier.fit(train_df[feature_names], train_df["age_group"])
        adult_train_df = train_df[train_df["label"].isin(["adult_male", "adult_female"])].reset_index(drop=True)
        child_train_df = train_df[train_df["label"].isin(["child_boy", "child_girl"])].reset_index(drop=True)
        adult_classifier.fit(adult_train_df[feature_names], adult_train_df["label"])
        child_classifier.fit(child_train_df[feature_names], child_train_df["label"])

        file_prediction_rows: List[Dict[str, object]] = []
        window_prediction_tables: List[pd.DataFrame] = []
        for _, group in test_df.groupby("path", sort=True):
            result_row, window_pred_df = self._predict_file_from_windows(
                group.reset_index(drop=True),
                age_classifier=age_classifier,
                adult_classifier=adult_classifier,
                child_classifier=child_classifier,
            )
            file_prediction_rows.append(result_row)
            window_prediction_tables.append(window_pred_df)

        file_predictions_df = pd.DataFrame(file_prediction_rows)
        window_predictions_df = pd.concat(window_prediction_tables, ignore_index=True) if window_prediction_tables else pd.DataFrame()

        y_true = file_predictions_df["label"].astype(str)
        y_pred = file_predictions_df["predicted_label"].astype(str)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "evaluation_level": "file",
            "train_files": int(len(train_files_df)),
            "test_files": int(len(test_files_df)),
            "train_windows": int(len(train_df)),
            "test_windows": int(len(test_df)),
        }
        report_text = classification_report(y_true, y_pred, labels=FOUR_CLASS_LABELS, digits=4)
        report_dict = classification_report(y_true, y_pred, labels=FOUR_CLASS_LABELS, digits=4, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        cm = confusion_matrix(y_true, y_pred, labels=FOUR_CLASS_LABELS)
        cm_df = pd.DataFrame(cm, index=FOUR_CLASS_LABELS, columns=FOUR_CLASS_LABELS)

        print("\nAccuracy (file-level):", f"{metrics['accuracy']:.4f}")
        print("Macro F1 (file-level):", f"{metrics['macro_f1']:.4f}")
        print("Weighted F1 (file-level):", f"{metrics['weighted_f1']:.4f}")
        print("\nClassification report:\n", report_text)
        print("Confusion matrix:\n", cm_df.to_string())

        self._save_feature_tables(all_df, train_df, test_df, feature_names, skipped_df)
        self._save_plots(train_df, test_df, feature_names)
        model_path = self._save_model_artifacts(
            age_classifier=age_classifier,
            adult_classifier=adult_classifier,
            child_classifier=child_classifier,
            feature_cols=feature_names,
            file_predictions_df=file_predictions_df,
            window_predictions_df=window_predictions_df,
            metrics=metrics,
            report_text=report_text,
            report_df=report_df,
            cm_df=cm_df,
            train_df=train_df,
            test_df=test_df,
        )
        self._write_root_summary(all_df, train_df, test_df, metrics, report_text, cm_df, skipped_df)

        print(f"\nГотово. Модель сохранена в: {model_path}")
        print(f"Таблицы сохранены в: {self.tables_dir}")
        print(f"Графики сохранены в: {self.plots_dir}")


def main() -> None:
    trainer = VoiceRandomForestTrainer(
        metadata_csv=Path(METADATA_CSV),
        class_folders=TRAIN_CLASS_FOLDERS,
        output_dir=Path(OUTPUT_DIR),
        audio_config=AudioConfig(),
        train_config=TrainConfig(),
    )
    trainer.run()


if __name__ == "__main__":
    main()
