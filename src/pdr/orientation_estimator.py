import numpy as np
import pandas as pd

from src.const import ANGLE, GYRO_X, TIMESTAMP
from src.utils.data_preprocessor import match_data


class OrientationEstimator:
    def __init__(self, drift_correction_factor: float = 0) -> None:
        self.drift_correction_factor = drift_correction_factor

    def __calculate_full_orientation(self, gyro_data: pd.DataFrame) -> pd.DataFrame:
        # 角速度を積分して角度を計算
        orientation = pd.DataFrame()
        orientation[TIMESTAMP] = gyro_data[TIMESTAMP]
        orientation[ANGLE] = np.cumsum(
            gyro_data[GYRO_X]
            * np.diff(gyro_data[TIMESTAMP], prepend=gyro_data[TIMESTAMP].iloc[0]),
        )

        # ドリフト補正
        time_elapsed = gyro_data[TIMESTAMP] - gyro_data[TIMESTAMP].iloc[0]
        orientation[ANGLE] -= (
            self.drift_correction_factor * time_elapsed * orientation[ANGLE].iloc[-1]
        )

        return orientation

    def estimate_step_orientations(
        self,
        gyro_data: pd.DataFrame,
        step_times: pd.Series,
    ) -> pd.DataFrame:
        # 全体の方向を計算
        full_orientation = self.__calculate_full_orientation(gyro_data)

        # 歩行ステップ時の方向データを抽出
        return match_data(full_orientation, step_times)
