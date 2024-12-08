from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from palkia.config import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    ANGLE,
    DEFAULT_STEP_LENGTH,
    GYRO_X,
    STEP_LENGTH,
    TIMESTAMP,
)
from palkia.core.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.core.sensor.data_preprocessor import match_data


class StepEstimator:
    def __init__(
        self,
        peak_threshold: float = 12,
        std_factor: float = 0.6,  # 標準偏差の係数
        min_step_time: float = 0.4,  # 最小ステップ間隔[秒]
        smoothing_time: float = 0.25,  # 平滑化時間[秒]
        step_length: float = DEFAULT_STEP_LENGTH,
        step_length_model_path: str | None = None,
        sampling_rate: float = 100,
    ) -> None:
        self.std_factor = std_factor
        self.peak_threshold = peak_threshold
        self.min_peak_distance = int(min_step_time * sampling_rate)
        self.smoothing_window = int(smoothing_time * sampling_rate)
        self.step_length = step_length
        self.step_length_model_path = step_length_model_path
        self.step_length_model: Any = None
        self.sampling_rate = sampling_rate
        if step_length_model_path:
            self._load_model(step_length_model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            self.step_length_model = joblib.load(model_path)
        except Exception as e:
            msg = f"Failed to load model from {model_path}: {e!s}"
            raise ValueError(msg) from e

    def _calculate_adaptive_threshold(self, acc_smoothed: np.ndarray) -> float:
        mean_acc = np.mean(acc_smoothed)
        std_acc = np.std(acc_smoothed)

        # ウィンドウサイズに応じて標準偏差の係数を調整
        adjusted_factor = self.std_factor * np.sqrt(25 / self.smoothing_window)

        return mean_acc + adjusted_factor * std_acc

    def estimate_steps(
        self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame
    ) -> pd.DataFrame:
        step_times = self.detect_step_times(acc_data)
        step_lengths = self._estimate_step_lengths(acc_data, gyro_data, step_times)
        return pd.DataFrame({TIMESTAMP: step_times, STEP_LENGTH: step_lengths})

    def detect_step_times(self, acc_data: pd.DataFrame) -> np.ndarray:
        acc_norm = self._calculate_acceleration_norm(acc_data)
        acc_smoothed = self._smooth_acceleration(acc_norm)

        adaptive_threshold = self._calculate_adaptive_threshold(acc_smoothed)

        peaks, _ = find_peaks(
            acc_smoothed,
            height=adaptive_threshold,
            distance=self.min_peak_distance,
        )
        return acc_data.iloc[peaks][TIMESTAMP].to_numpy()

    def _calculate_acceleration_norm(self, acc_data: pd.DataFrame) -> np.ndarray:
        return np.sqrt(
            acc_data[ACC_X] ** 2 + acc_data[ACC_Y] ** 2 + acc_data[ACC_Z] ** 2
        )

    def _create_gaussian_kernel(self, sigma: float) -> np.ndarray:
        # カーネルサイズを計算(6σルール)
        kernel_size = int(6 * sigma * self.sampling_rate)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 奇数にする

        # カーネルの中心からの距離を計算
        x = np.linspace(-3, 3, kernel_size)

        # ガウシアンカーネルを計算
        kernel = np.exp(-(x**2) / (2))

        # カーネルを正規化
        return kernel / kernel.sum()

    def _smooth_acceleration(self, acc_norm: np.ndarray) -> np.ndarray:
        # ガウシアンカーネルを生成
        kernel = self._create_gaussian_kernel(0.1)

        # エッジ処理のためにデータを拡張
        pad_width = len(kernel) // 2
        acc_padded = np.pad(acc_norm, (pad_width, pad_width), mode="edge")

        # ガウス畳み込みを適用
        return np.convolve(acc_padded, kernel, mode="valid")

    def _estimate_step_lengths(
        self, acc_data: pd.DataFrame, gyro_data: pd.DataFrame, step_times: np.ndarray
    ) -> np.ndarray:
        if self.step_length_model is None:
            return np.full(len(step_times), self.step_length)

        step_timings_acc = match_data(acc_data, pd.Series(step_times))
        step_timings_orientations = OrientationEstimator().estimate_step_orientations(
            gyro_data, pd.Series(step_times)
        )

        # acc_norm_smoothed = self._smooth_acceleration(
        #     self._calculate_acceleration_norm(step_timings_acc)
        # )

        acc_norm = self._calculate_acceleration_norm(step_timings_acc)

        orientations_diff = self._calculate_orientations_difference(
            step_timings_orientations
        )

        return self._predict_step_lengths(acc_norm, orientations_diff)

    def _calculate_gyro_difference(self, gyro_data: pd.DataFrame) -> np.ndarray:
        gyro_diff = np.diff(gyro_data[GYRO_X])
        return np.insert(gyro_diff, 0, np.mean(gyro_diff))

    def _calculate_orientations_difference(
        self, orientations_data: pd.DataFrame
    ) -> np.ndarray:
        orientations_diff = np.diff(orientations_data[ANGLE])
        return np.insert(orientations_diff, 0, np.mean(orientations_diff))

    def _predict_step_lengths(
        self, acc_norm_smoothed: np.ndarray, gyro_diff: np.ndarray
    ) -> np.ndarray:
        if self.step_length_model is None:
            msg = "Step length model is not loaded."
            raise ValueError(msg)

        data = np.column_stack((acc_norm_smoothed, gyro_diff))
        return self.step_length_model.predict(data)

    # orientationを使用してstep_lengthを推定する関数群

    def estimate_steps_from_orientation(
        self, acc_data: pd.DataFrame, orientation_data: pd.DataFrame
    ) -> pd.DataFrame:
        step_times = self.detect_step_times(acc_data)
        step_lengths = self._estimate_step_lengths_from_orientation(
            acc_data, orientation_data, step_times
        )
        return pd.DataFrame({TIMESTAMP: step_times, STEP_LENGTH: step_lengths})

    def _estimate_step_lengths_from_orientation(
        self,
        acc_data: pd.DataFrame,
        orientation_data: pd.DataFrame,
        step_times: np.ndarray,
    ) -> np.ndarray:
        if self.step_length_model is None:
            return np.full(len(step_times), self.step_length)

        step_timings_acc = match_data(acc_data, pd.Series(step_times))
        step_times_orientation = match_data(orientation_data, pd.Series(step_times))
        step_timings_norm = self._calculate_acceleration_norm(step_timings_acc)

        orientations_diff = self._calculate_orientations_difference(
            step_times_orientation
        )

        return self._predict_step_lengths(step_timings_norm, orientations_diff)
