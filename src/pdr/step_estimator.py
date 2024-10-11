from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.const import ACC_X, ACC_Y, ACC_Z, DEFAULT_STEP_LENGTH, STEP_LENGTH, TIMESTAMP


class Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class StepEstimator:
    def __init__(
        self,
        peak_threshold: float = 12,
        window_size: int = 10,
        step_length: float = DEFAULT_STEP_LENGTH,
        step_length_model: Predictor | None = None,
    ) -> None:
        self.peak_threshold = peak_threshold
        self.window_size = window_size
        self.step_length_model = step_length_model
        self.step_length = step_length

    def estimate_steps(
        self, acc_data: pd.DataFrame, step_orientations: pd.DataFrame
    ) -> pd.DataFrame:
        step_times = self.detect_step_times(acc_data)
        step_lengths = self.__estimate_step_lengths(acc_data, step_orientations)

        return pd.DataFrame({TIMESTAMP: step_times, STEP_LENGTH: step_lengths})

    def detect_step_times(self, acc_data: pd.DataFrame) -> np.ndarray:
        acc_magnitude = np.sqrt(
            acc_data[ACC_X] ** 2 + acc_data[ACC_Y] ** 2 + acc_data[ACC_Z] ** 2,
        )
        acc_smoothed = pd.Series(acc_magnitude).rolling(window=self.window_size).mean()
        peaks, _ = find_peaks(
            acc_smoothed,
            height=self.peak_threshold,
            distance=self.window_size,
        )
        return acc_data.iloc[peaks][TIMESTAMP].to_numpy()

    def __estimate_step_lengths(
        self, acc_data: pd.DataFrame, step_orientations: pd.DataFrame
    ) -> np.ndarray:
        if self.step_length_model is None:
            return np.full(len(step_orientations), self.step_length)

        features = self.__extract_features(acc_data, step_orientations)
        return self.step_length_model.predict(features)

    def __extract_features(
        self, _: pd.DataFrame, step_orientations: pd.DataFrame
    ) -> np.ndarray:
        return np.zeros((len(step_orientations), 10))  # ダミーの特徴量
