from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# НАСТРОЙКИ ПРОЕКТА
# ============================================================

MODEL_BUNDLE_PATH: str = (
    r"/home/test/Рабочий стол/Классификатор пола/random_forest_window/"
    r"random_forest_window_results/model/voice_random_forest_model_bundle.joblib"
)
INPUT_AUDIO_PATH: str = r"/home/test/Рабочий стол/Классификатор пола/данные/2"
OUTPUT_DIR: str = (
    r"/home/test/Рабочий стол/Классификатор пола/random_forest_window/"
    r"random_forest_window_inference_results"
)
INFERENCE_METADATA_CSV: Optional[str] = None
DEFAULT_FOUR_CLASS_LABELS: List[str] = ["adult_male", "adult_female", "child_boy", "child_girl"]


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
    probability_aggregation: str = "mean"


class PathUtils:
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def collect_audio_files(input_path: Path) -> List[Path]:
        if input_path.is_file():
            return [input_path.resolve()]
        if input_path.is_dir():
            return sorted(
                [
                    file.resolve()
                    for file in input_path.rglob("*")
                    if file.is_file() and file.suffix.lower() in PathUtils.AUDIO_EXTS
                ]
            )
        raise FileNotFoundError(f"Не найден путь: {input_path}")


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

    def extract_windows(self, audio_path: Path) -> pd.DataFrame:
        y = self.load_audio(audio_path)
        rows: List[Dict[str, float]] = []
        for window_idx, start_sec, window in self.split_into_windows(y):
            feats = self.extract_from_signal(window)
            feats["window_idx"] = int(window_idx)
            feats["window_start_sec"] = float(start_sec)
            rows.append(feats)
        if not rows:
            raise RuntimeError("Не удалось извлечь ни одного окна")
        return pd.DataFrame(rows)


class StagePredictor:
    def __init__(self, stage_bundle: Dict[str, object]) -> None:
        self.scaler = stage_bundle["scaler"]
        self.model = stage_bundle["model"]
        self.feature_names = list(stage_bundle["feature_names"])
        self.classes = list(stage_bundle["classes"])

    def predict_proba_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.transform(X[self.feature_names].values)
        probs = self.model.predict_proba(X_scaled)
        return pd.DataFrame(probs, columns=self.classes, index=X.index)


class CascadeRandomForestPredictor:
    def __init__(self, model_bundle_path: Path) -> None:
        self.model_bundle_path = model_bundle_path.resolve()

        if self.model_bundle_path.suffix.lower() != ".joblib":
            raise ValueError("Недопустимый формат файла модели.")
        if not self.model_bundle_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_bundle_path}")

        bundle = joblib.load(self.model_bundle_path)  # nosemgrep: python-unsafe-joblib-load
        if "cascade" not in bundle:
            raise KeyError(
                "В указанном bundle нет ключа 'cascade'. Проверьте, что путь ведет к новой каскадной модели."
            )
        self.audio_config = AudioConfig(**bundle["audio_config"])
        self.feature_names = list(bundle["feature_names"])
        self.classes = list(bundle.get("classes", DEFAULT_FOUR_CLASS_LABELS))
        cascade = bundle["cascade"]
        self.age_predictor = StagePredictor(cascade["age_group"])
        self.adult_predictor = StagePredictor(cascade["adult_gender"])
        self.child_predictor = StagePredictor(cascade["child_gender"])
        self.extractor = FeatureExtractor(self.audio_config)

    def _select_informative_windows(self, windows_df: pd.DataFrame) -> pd.DataFrame:
        df = windows_df.copy().reset_index(drop=True)
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
        selected_idx = candidates.head(int(self.audio_config.max_windows_per_file_infer)).index
        df.loc[selected_idx, "is_selected"] = 1
        selected = df.loc[df["is_selected"] == 1].copy()
        if selected.empty:
            selected = df.sort_values(by=["window_idx"]).head(1).copy()
            selected["is_selected"] = 1
        return selected.reset_index(drop=True)

    def _aggregate_probabilities(self, combined: pd.DataFrame) -> pd.Series:
        mode = str(self.audio_config.probability_aggregation).lower().strip()
        if mode == "median":
            return combined.median(axis=0)
        return combined.mean(axis=0)

    def predict_file(self, audio_path: Path) -> Tuple[Dict[str, object], pd.DataFrame]:
        windows_df = self.extractor.extract_windows(audio_path)
        selected_windows = self._select_informative_windows(windows_df)
        X = selected_windows[self.feature_names].copy()

        age_probs = self.age_predictor.predict_proba_df(X)
        adult_probs = self.adult_predictor.predict_proba_df(X)
        child_probs = self.child_predictor.predict_proba_df(X)

        combined = pd.DataFrame(index=X.index)
        combined["adult_male"] = age_probs["adult"] * adult_probs["adult_male"]
        combined["adult_female"] = age_probs["adult"] * adult_probs["adult_female"]
        combined["child_boy"] = age_probs["child"] * child_probs["child_boy"]
        combined["child_girl"] = age_probs["child"] * child_probs["child_girl"]

        agg_probs = self._aggregate_probabilities(combined)
        predicted_label = str(agg_probs.idxmax())

        result: Dict[str, object] = {
            "path": str(audio_path),
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

        window_predictions = selected_windows[["window_idx", "window_start_sec", "window_score", "is_selected"]].copy()
        window_predictions.insert(0, "path", str(audio_path))
        window_predictions["window_predicted_label"] = combined.idxmax(axis=1).astype(str)
        window_predictions["proba_male"] = combined["adult_male"].values
        window_predictions["proba_female"] = combined["adult_female"].values
        window_predictions["proba_boy"] = combined["child_boy"].values
        window_predictions["proba_girl"] = combined["child_girl"].values
        window_predictions["stage1_adult"] = age_probs["adult"].values
        window_predictions["stage1_child"] = age_probs["child"].values
        return result, window_predictions


class InferenceMetadata:
    def __init__(self, metadata_csv: Path) -> None:
        self.metadata_csv = metadata_csv.resolve()
        self.table = self._load_table()

    def _load_table(self) -> pd.DataFrame:
        meta = pd.read_csv(self.metadata_csv)
        required_columns = {"path", "label"}
        if not required_columns.issubset(meta.columns):
            raise ValueError("В metadata.csv должны быть столбцы: path и label")
        meta = meta.copy()
        meta["resolved_path"] = meta["path"].map(
            lambda p: str((self.metadata_csv.parent / str(p)).resolve())
            if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve())
        )
        return meta[["resolved_path", "label"]].drop_duplicates().reset_index(drop=True)

    def get_label_map(self) -> Dict[str, str]:
        return dict(zip(self.table["resolved_path"], self.table["label"]))


class VoiceRandomForestInferenceRunner:
    def __init__(
        self,
        model_bundle_path: Path,
        input_audio_path: Path,
        output_dir: Path,
        inference_metadata_csv: Optional[Path] = None,
    ) -> None:
        self.model_bundle_path = model_bundle_path.resolve()
        self.input_audio_path = input_audio_path.resolve()
        self.output_dir = output_dir.resolve()
        self.inference_metadata_csv = inference_metadata_csv.resolve() if inference_metadata_csv else None
        self.predictor = CascadeRandomForestPredictor(self.model_bundle_path)

    def run(self) -> None:
        PathUtils.ensure_dir(self.output_dir)
        audio_files = PathUtils.collect_audio_files(self.input_audio_path)
        if not audio_files:
            raise RuntimeError("Не найдено ни одного аудиофайла для классификации")

        results: List[Dict[str, object]] = []
        window_tables: List[pd.DataFrame] = []
        skipped: List[Dict[str, str]] = []
        for idx, file_path in enumerate(audio_files, start=1):
            try:
                row, window_df = self.predictor.predict_file(file_path)
                results.append(row)
                window_tables.append(window_df)
                if idx == 1 or idx % 25 == 0 or idx == len(audio_files):
                    print(f"[INFO] Классифицировано файлов: {idx}/{len(audio_files)}")
            except Exception as exc:
                skipped.append({"path": str(file_path), "reason": str(exc)})
                print(f"[SKIP] {file_path}: {exc}")

        if not results:
            raise RuntimeError("Не удалось классифицировать ни одного файла")

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
        if window_tables:
            pd.concat(window_tables, ignore_index=True).to_csv(
                self.output_dir / "window_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if skipped:
            pd.DataFrame(skipped).to_csv(self.output_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")

        summary_lines = [f"Всего успешно классифицировано: {len(results_df)}"]

        if self.inference_metadata_csv is not None:
            label_map = InferenceMetadata(self.inference_metadata_csv).get_label_map()
            results_df["true_label"] = results_df["path"].map(label_map)
            eval_df = results_df.dropna(subset=["true_label"]).copy()
            results_df.to_csv(self.output_dir / "predictions_with_labels.csv", index=False, encoding="utf-8-sig")
            if not eval_df.empty:
                y_true = eval_df["true_label"].astype(str)
                y_pred = eval_df["predicted_label"].astype(str)
                labels = list(self.predictor.classes)
                accuracy = float(accuracy_score(y_true, y_pred))
                macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
                weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
                report_text = classification_report(y_true, y_pred, labels=labels, digits=4)
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                cm_df.to_csv(self.output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
                with open(self.output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
                    f.write(report_text)
                with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "accuracy": accuracy,
                            "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1,
                            "evaluated_files": int(len(eval_df)),
                            "evaluation_level": "file",
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                summary_lines.extend([
                    "",
                    f"Оценено файлов по metadata.csv: {len(eval_df)}",
                    f"Accuracy: {accuracy:.4f}",
                    f"Macro F1: {macro_f1:.4f}",
                    f"Weighted F1: {weighted_f1:.4f}",
                    "",
                    "Classification report:",
                    report_text,
                    "",
                    "Confusion matrix:",
                    cm_df.to_string(),
                ])
                print("\nAccuracy:", f"{accuracy:.4f}")
                print("Macro F1:", f"{macro_f1:.4f}")
                print("Weighted F1:", f"{weighted_f1:.4f}")
            else:
                summary_lines.append("\nВ metadata.csv не найдено совпадающих путей для расчета метрик.")

        with open(self.output_dir / "INFERENCE_SUMMARY.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nГотово. Результаты сохранены в: {self.output_dir}")


def main() -> None:
    runner = VoiceRandomForestInferenceRunner(
        model_bundle_path=Path(MODEL_BUNDLE_PATH),
        input_audio_path=Path(INPUT_AUDIO_PATH),
        output_dir=Path(OUTPUT_DIR),
        inference_metadata_csv=Path(INFERENCE_METADATA_CSV) if INFERENCE_METADATA_CSV else None,
    )
    runner.run()


if __name__ == "__main__":
    main()
