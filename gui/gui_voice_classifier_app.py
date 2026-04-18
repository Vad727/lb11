from __future__ import annotations

import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.signal import lfilter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_BUNDLE_FILENAME = "voice_random_forest_model_bundle.joblib"
DEFAULT_FOUR_CLASS_LABELS: List[str] = ["adult_male", "adult_female", "child_boy", "child_girl"]
CLASS_NAME_RU: Dict[str, str] = {
    "adult_male": "Взрослый мужчина",
    "adult_female": "Взрослая женщина",
    "child_boy": "Мальчик",
    "child_girl": "Девочка",
}


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
            raise KeyError("В указанном файле модели нет ключа 'cascade'.")

        raw_audio_config = dict(bundle.get("audio_config", {}))
        allowed_keys = set(AudioConfig.__dataclass_fields__.keys())
        filtered_audio_config = {k: v for k, v in raw_audio_config.items() if k in allowed_keys}
        self.audio_config = AudioConfig(**filtered_audio_config)
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

    def predict_file(self, audio_path: Path) -> Dict[str, object]:
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

        return {
            "path": str(audio_path),
            "predicted_label": predicted_label,
            "predicted_label_ru": CLASS_NAME_RU.get(predicted_label, predicted_label),
            "n_windows": int(len(selected_windows)),
            "probabilities": {
                "adult_male": float(agg_probs["adult_male"]),
                "adult_female": float(agg_probs["adult_female"]),
                "child_boy": float(agg_probs["child_boy"]),
                "child_girl": float(agg_probs["child_girl"]),
            },
        }


class VoiceClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Классификатор голоса")
        self.root.geometry("650x420")
        self.root.minsize(620, 390)

        self.audio_path: Path | None = None
        self.predictor: CascadeRandomForestPredictor | None = None

        self._build_ui()
        self._load_model_on_startup()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=16)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)

        title = ttk.Label(main, text="Классификатор аудиофайлов", font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.model_status_var = tk.StringVar(value="Модель: поиск...")
        ttk.Label(main, textvariable=self.model_status_var).grid(row=1, column=0, sticky="w", pady=(0, 12))

        file_frame = ttk.LabelFrame(main, text="Аудиофайл", padding=12)
        file_frame.grid(row=2, column=0, sticky="ew")
        file_frame.columnconfigure(0, weight=1)

        self.file_var = tk.StringVar(value="Файл не выбран")
        ttk.Label(file_frame, textvariable=self.file_var, wraplength=540).grid(row=0, column=0, sticky="w")
        ttk.Button(file_frame, text="Загрузить аудиофайл", command=self.select_audio_file).grid(row=1, column=0, sticky="w", pady=(10, 0))

        action_frame = ttk.Frame(main)
        action_frame.grid(row=3, column=0, sticky="ew", pady=14)
        self.analyze_button = ttk.Button(action_frame, text="Запустить анализ", command=self.run_analysis, state="disabled")
        self.analyze_button.grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Выберите аудиофайл")
        ttk.Label(main, textvariable=self.status_var, foreground="#444444").grid(row=4, column=0, sticky="w", pady=(0, 12))

        result_frame = ttk.LabelFrame(main, text="Результат", padding=12)
        result_frame.grid(row=5, column=0, sticky="nsew")
        result_frame.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)

        self.result_var = tk.StringVar(value="—")
        ttk.Label(result_frame, text="Класс:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(result_frame, textvariable=self.result_var, font=("Segoe UI", 13, "bold")).grid(row=1, column=0, sticky="w", pady=(2, 10))

        self.prob_text = tk.Text(result_frame, height=10, wrap="word", state="disabled", font=("Consolas", 10))
        self.prob_text.grid(row=2, column=0, sticky="nsew")
        result_frame.rowconfigure(2, weight=1)

    def _get_runtime_base_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def _candidate_model_paths(self) -> List[Path]:
        base = self._get_runtime_base_dir()
        return [
            base / MODEL_BUNDLE_FILENAME,
            base / "model" / MODEL_BUNDLE_FILENAME,
        ]

    def _load_model_on_startup(self) -> None:
        for path in self._candidate_model_paths():
            if path.exists():
                try:
                    self.predictor = CascadeRandomForestPredictor(path)
                    self.model_status_var.set(f"Модель: {path}")
                    self._refresh_analyze_button_state()
                    return
                except Exception as exc:
                    self.model_status_var.set(f"Ошибка загрузки модели: {exc}")
                    return
        self.model_status_var.set(
            "Модель не найдена. Положите voice_random_forest_model_bundle.joblib рядом с exe или в папку model."
        )

    def _refresh_analyze_button_state(self) -> None:
        enabled = self.predictor is not None and self.audio_path is not None
        self.analyze_button.configure(state="normal" if enabled else "disabled")

    def select_audio_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудиофайлы", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("Все файлы", "*.*"),
            ],
        )
        if not filename:
            return
        self.audio_path = Path(filename)
        self.file_var.set(str(self.audio_path))
        self.status_var.set("Файл выбран. Можно запускать анализ.")
        self._refresh_analyze_button_state()

    def run_analysis(self) -> None:
        if self.predictor is None:
            messagebox.showerror("Ошибка", "Модель не загружена.")
            return
        if self.audio_path is None:
            messagebox.showwarning("Внимание", "Сначала выберите аудиофайл.")
            return

        self.analyze_button.configure(state="disabled")
        self.status_var.set("Идёт анализ...")
        self.result_var.set("—")
        self._set_prob_text("")

        thread = threading.Thread(target=self._analyze_worker, daemon=True)
        thread.start()

    def _analyze_worker(self) -> None:
        try:
            assert self.predictor is not None
            assert self.audio_path is not None
            result = self.predictor.predict_file(self.audio_path)
            self.root.after(0, lambda: self._on_analysis_success(result))
        except Exception as exc:
            self.root.after(0, lambda: self._on_analysis_error(exc))

    def _on_analysis_success(self, result: Dict[str, object]) -> None:
        label_ru = str(result["predicted_label_ru"])
        probs: Dict[str, float] = result["probabilities"]  # type: ignore[assignment]
        n_windows = int(result["n_windows"])

        self.result_var.set(label_ru)
        lines = [f"Использовано окон: {n_windows}", "", "Вероятности по классам:"]
        for key in ["adult_male", "adult_female", "child_boy", "child_girl"]:
            lines.append(f"- {CLASS_NAME_RU.get(key, key)}: {probs[key] * 100:.2f}%")
        self._set_prob_text("\n".join(lines))
        self.status_var.set("Анализ завершён")
        self._refresh_analyze_button_state()

    def _on_analysis_error(self, exc: Exception) -> None:
        self.status_var.set("Ошибка анализа")
        self._refresh_analyze_button_state()
        messagebox.showerror("Ошибка", str(exc))

    def _set_prob_text(self, text: str) -> None:
        self.prob_text.configure(state="normal")
        self.prob_text.delete("1.0", tk.END)
        self.prob_text.insert("1.0", text)
        self.prob_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    app = VoiceClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
