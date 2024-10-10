import numpy as np
import pandas as pd

from .pdr_estimator import PDREstimator


class ThreeDimensionalEstimator:
    def __init__(self, pdr_estimator: PDREstimator) -> None:
        self.pdr_estimator = pdr_estimator

    def estimate_3d_trajectory(
        self,
        acc_data: pd.DataFrame,
        gyro_data: pd.DataFrame,
        baro_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # 2D軌跡の推定
        trajectory_2d = self.pdr_estimator.estimate_trajectory(acc_data, gyro_data)

        # 気圧データから高度を推定
        height = self.__estimate_height_from_pressure(baro_data)

        # 3D軌跡の生成
        trajectory_3d = trajectory_2d.copy()
        trajectory_3d["z"] = np.interp(
            trajectory_3d["timestamp"],
            baro_data["timestamp"],
            height,
        )

        return trajectory_3d

    def __estimate_height_from_pressure(self, baro_data: pd.DataFrame) -> np.ndarray:
        # 気圧から高度への変換 簡易実装
        pressure_sea_level = 1013.25  # hPa
        height = 44330 * (
            1 - (baro_data["pressure"] / pressure_sea_level) ** (1 / 5.255)
        )

        return height.to_numpy()
