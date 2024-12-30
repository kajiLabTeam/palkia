# palkia/core/positioning/correction/trajectory_correctors_builder.py

from typing import Optional

import pandas as pd

from palkia.core.map import FloorMap
from palkia.core.positioning.pdr import PDREstimator

from .ble_correction import BLECorrector
from .drift import DriftCorrector
from .map_matching import MapMatcher
from .trajectory_corrector import TrajectoryCorrector


class TrajectoryCorrectorsBuilder:
    def __init__(self, pdr_estimator: PDREstimator) -> None:
        self.pdr_estimator = pdr_estimator
        self._floor_map: FloorMap
        self._gt_data: pd.DataFrame | None = None
        self._ble_data: pd.DataFrame | None = None

    def with_floor_map(self, floor_map: FloorMap) -> "TrajectoryCorrectorsBuilder":
        self._floor_map = floor_map
        return self

    def with_ground_truth(self, gt_data: pd.DataFrame) -> "TrajectoryCorrectorsBuilder":
        self._gt_data = gt_data
        return self

    def with_ble_data(self, ble_data: pd.DataFrame) -> "TrajectoryCorrectorsBuilder":
        self._ble_data = ble_data
        return self

    def build(self) -> TrajectoryCorrector:
        drift_corrector = DriftCorrector({}, self.pdr_estimator, self._gt_data)
        map_matcher = MapMatcher({}, self.pdr_estimator, self._floor_map)
        ble_corrector = BLECorrector()

        return TrajectoryCorrector(
            pdr_estimator=self.pdr_estimator,
            drift_corrector=drift_corrector,
            map_matcher=map_matcher,
            ble_corrector=ble_corrector,
        )
