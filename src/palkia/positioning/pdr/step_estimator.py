from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from palkia.const import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    DEFAULT_STEP_LENGTH,
    GYRO_X,
    STEP_LENGTH,
    TIMESTAMP,
)
from palkia.utils.data_preprocessor import match_data


class StepEstimator:
    def __init__(
        self,
        peak_threshold: float = 12,
        window_size: int = 10,
        step_length: float = DEFAULT_STEP_LENGTH,
        step_length_model_path: str | None = None,
    ) -> None:
        self.peak_threshold = peak_threshold
        self.window_size = window_size
        self.step_length = step_length
        self.step_length_model_path = step_length_model_path
        self.step_length_model: Any = None
        if step_length_model_path:
            self._load_model(step_length_model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            self.step_length_model = joblib.load(model_path)
        except Exception as e:
            msg = f"Failed to load model from {model_path}: {e!s}"
            raise ValueError(msg) from e

    def estimate_steps(
        self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame
    ) -> pd.DataFrame:
        step_times = self.detect_step_times(acc_data)
        step_lengths = self._estimate_step_lengths(acc_data, gyro_data, step_times)
        return pd.DataFrame({TIMESTAMP: step_times, STEP_LENGTH: step_lengths})

    def detect_step_times(self, acc_data: pd.DataFrame) -> np.ndarray:
        acc_norm = self._calculate_acceleration_norm(acc_data)
        acc_smoothed = self._smooth_acceleration(acc_norm)
        peaks, _ = find_peaks(
            acc_smoothed,
            height=self.peak_threshold,
            distance=self.window_size,
        )
        return acc_data.iloc[peaks][TIMESTAMP].to_numpy()

    def _calculate_acceleration_norm(self, acc_data: pd.DataFrame) -> np.ndarray:
        return np.sqrt(
            acc_data[ACC_X] ** 2 + acc_data[ACC_Y] ** 2 + acc_data[ACC_Z] ** 2
        )

    # 平滑化
    def _smooth_acceleration(self, acc_norm: np.ndarray) -> np.ndarray:
        kernel = np.ones(self.window_size) / self.window_size
        return np.convolve(acc_norm, kernel, mode="same")

    def _estimate_step_lengths(
        self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame, step_times: np.ndarray
    ) -> np.ndarray:
        if self.step_length_model is None:
            return np.full(len(step_times), self.step_length)

        acc_data_step_timings = match_data(acc_data, pd.Series(step_times))
        gyro_data_step_timings = match_data(gyro_data, pd.Series(step_times))

        acc_norm_smoothed = self._smooth_acceleration(
            self._calculate_acceleration_norm(acc_data_step_timings)
        )
        gyro_diff = self._calculate_gyro_difference(gyro_data_step_timings)

        return self._predict_step_lengths(acc_norm_smoothed, gyro_diff)

    def _calculate_gyro_difference(self, gyro_data: pd.DataFrame) -> np.ndarray:
        gyro_diff = np.diff(gyro_data[GYRO_X])
        return np.insert(gyro_diff, 0, np.mean(gyro_diff))

    def _predict_step_lengths(
        self, acc_norm_smoothed: np.ndarray, gyro_diff: np.ndarray
    ) -> np.ndarray:
        if self.step_length_model is None:
            msg = "Step length model is not loaded."
            raise ValueError(msg)

        data = np.column_stack((acc_norm_smoothed, gyro_diff))
        return self.step_length_model.predict(data)
