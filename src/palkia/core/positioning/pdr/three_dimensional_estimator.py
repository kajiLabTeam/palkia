# palkia/positioning/pdr/three_dimensional_estimator.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from palkia.config import PRESSURE
from palkia.config.column_name import TIMESTAMP
from palkia.core.positioning.floor_identification import FloorIdentifier, FloorInfo

if TYPE_CHECKING:
    from palkia.core.map.floor_map import FloorMap
    from palkia.core.positioning.correction.trajectory_corrector import (
        TrajectoryCorrector,
    )
    from palkia.core.positioning.pdr import PDREstimator


# TODO: 軌跡補正のために拡張する必要性がある(補正classを内部で持たせてあげるといいかも?)
class ThreeDimensionalEstimator:
    def __init__(
        self,
        trajectory_corrector: TrajectoryCorrector,
    ) -> None:
        self.trajectory_corrector = trajectory_corrector

    def estimate_3d_trajectory_with_floors(
        self,
        baro_data: pd.DataFrame,
        floor_maps: dict[int, FloorMap],
    ) -> dict[int, FloorInfo]:
        """3次元の軌跡を推定し、階層情報を付加する.

        Args:
        ----
            baro_data: 気圧センサーデータ
            floor_maps: 各階のフロアマップ

        Returns:
        -------
            Dict[int, FloorInfo]: 階層ごとの軌跡情報

        """
        # まず基本の軌跡を推定
        trajectory_2d = self.trajectory_corrector.estimate_and_correct_trajectory()

        # 気圧データから高度を推定
        height = self._estimate_height_from_pressure(baro_data)

        # 3D軌跡の生成(高度情報を追加)
        trajectory_3d = trajectory_2d.copy()
        trajectory_3d["z"] = np.interp(
            trajectory_3d[TIMESTAMP],
            baro_data[TIMESTAMP],
            height,
        )

        # 気圧データを軌跡データにマージ
        trajectory_with_pressure = pd.merge_asof(
            trajectory_3d,
            baro_data[[TIMESTAMP, PRESSURE]],
            on=TIMESTAMP,
            direction="nearest",
        )

        # 階層識別を実行
        return FloorIdentifier().identify_floors(
            baro_data=baro_data,
            trajectory=trajectory_with_pressure,
            floor_maps=floor_maps,
        )

    def _estimate_height_from_pressure(self, baro_data: pd.DataFrame) -> np.ndarray:
        """気圧から高度への変換(簡易実装)."""
        pressure_sea_level = 1013.25  # hPa
        height = 44330 * (1 - (baro_data[PRESSURE] / pressure_sea_level) ** (1 / 5.255))
        return height.to_numpy()
