import numpy as np
import pandas as pd

from src.const import ANGLE, GYRO_X, TIMESTAMP


class OrientationEstimator:
    def __init__(self, drift_correction_factor: float = 0.01):
        self.drift_correction_factor = drift_correction_factor

    def estimate_orientation(self, gyro_data: pd.DataFrame) -> pd.DataFrame:
        # 角速度を積分して角度を計算
        orientation = pd.DataFrame()
        orientation[TIMESTAMP] = gyro_data[TIMESTAMP]
        orientation[ANGLE] = np.cumsum(
            gyro_data[GYRO_X]
            * np.diff(gyro_data[TIMESTAMP], prepend=gyro_data[TIMESTAMP].iloc[0]),
        )

        # ドリフト補正
        # time_elapsed = gyro_data[TIMESTAMP] - gyro_data[TIMESTAMP].iloc[0]
        # orientation["angle"] -= (
        #     self.drift_correction_factor * time_elapsed * orientation["angle"].iloc[-1]
        # )

        return orientation
