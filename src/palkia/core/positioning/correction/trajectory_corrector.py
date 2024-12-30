from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from palkia.core.map import FloorMap
from palkia.core.positioning.pdr import (
    OrientationEstimator,
    StepEstimator,
    TrajectoryCalculator,
)

if TYPE_CHECKING:
    import pandas as pd

    from examples.main import FloorMap
    from palkia.core.positioning.pdr import (
        PDREstimator,
    )

    from .ble_correction import BLECorrector
    from .drift import DriftCorrector
    from .map_matching import MapMatcher


class TrajectoryCorrector:
    def __init__(
        self,
        pdr_estimator: PDREstimator,
        drift_corrector: DriftCorrector,
        map_matcher: MapMatcher,
        ble_corrector: BLECorrector,
    ) -> None:
        self.pdr_estimator = pdr_estimator
        self.drift_corrector = drift_corrector
        self.map_matcher = map_matcher
        self.ble_corrector = ble_corrector

    def estimate_and_correct_trajectory(
        self,
        ble_data: pd.DataFrame | None = None,
        floor_map: FloorMap | None = None,
    ) -> pd.DataFrame:
        """PDRを用いて軌跡を推定し、各種補正を適用する.

        Args:
        ----
            acc_data (pd.DataFrame): 加速度データ
            gyro_data (pd.DataFrame): ジャイロデータ
            ble_data (pd.DataFrame, optional): BLEデータ
            floor_map (Any, optional): フロアマップデータ

        Returns:
        -------
            pd.DataFrame: 推定・補正された軌跡

        """
        # 初期軌跡の計算
        trajectory = self.pdr_estimator.estimate_trajectory()
        # ドリフト補正
        remove_drift_trajectory = self.drift_corrector.correct()

        if ble_data is not None:
            trajectory = self.ble_corrector.correct_initial_direction(
                remove_drift_trajectory,
                ble_data,
            )
        elif floor_map is not None:
            trajectory = self.map_matcher.correct_initial_direction()

        if floor_map is not None:
            trajectory = self.map_matcher.correct_unwalkable_points(trajectory)

        return trajectory
