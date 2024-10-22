import numpy as np
import pandas as pd

from palkia.const import TIMESTAMP

from .pdr_estimator import PDREstimator


class ThreeDimensionalEstimator:
    def __init__(self, pdr_estimator: PDREstimator) -> None:
        self.pdr_estimator = pdr_estimator

    def estimate_3d_trajectory(
        self,
        baro_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # 2D軌跡の推定
        trajectory_2d = self.pdr_estimator.estimate_trajectory()
        # 気圧データから高度を推定
        height = self.__estimate_height_from_pressure(baro_data)
        # 3D軌跡の生成
        trajectory_3d = trajectory_2d.copy()
        trajectory_3d["z"] = np.interp(
            trajectory_3d[TIMESTAMP],
            baro_data[TIMESTAMP],
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
