import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.const import ACC_X, ACC_Y, ACC_Z, TIMESTAMP


class StepDetector:
    def __init__(self, peak_threshold: float = 12, window_size: int = 10) -> None:
        self.peak_threshold = peak_threshold
        self.window_size = window_size

    def detect_steps_ts(self, acc_data: pd.DataFrame) -> np.ndarray:
        # 加速度の大きさを計算
        acc_magnitude = np.sqrt(
            acc_data[ACC_X] ** 2 + acc_data[ACC_Y] ** 2 + acc_data[ACC_Z] ** 2,
        )

        # 移動平均を適用
        acc_smoothed = pd.Series(acc_magnitude).rolling(window=self.window_size).mean()

        # ピーク検出
        peaks, _ = find_peaks(
            acc_smoothed,
            height=self.peak_threshold,
            distance=self.window_size,
        )

        # 加速度のtimestampで返す
        return acc_data.iloc[peaks][TIMESTAMP].to_numpy()
