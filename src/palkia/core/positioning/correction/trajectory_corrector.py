from __future__ import annotations

from typing import TYPE_CHECKING

from palkia.core.map import FloorMap
from palkia.core.visualization.trajectory_plotter import plot_trajectory

from .drift import DriftCorrector
from .map_matching import MapMatcher
from .wireless_signal_corrector import WirelessSignalCorrector

if TYPE_CHECKING:
    import pandas as pd

    from examples.main import FloorMap
    from palkia.core.positioning.pdr import (
        PDREstimator,
    )


class TrajectoryCorrector:
    def __init__(
        self,
        pdr_estimator: PDREstimator,
        drift_corrector: DriftCorrector,
        map_matcher: MapMatcher | None,
        ble_corrector: WirelessSignalCorrector | None,
    ) -> None:
        self.pdr_estimator = pdr_estimator
        self.drift_corrector = drift_corrector
        self.map_matcher = map_matcher
        self.ble_corrector = ble_corrector

    def estimate_and_correct_trajectory(self) -> pd.DataFrame:
        trajectory = self.pdr_estimator.estimate_trajectory()
        trajectory = self.drift_corrector.correct_drift()

        if self.ble_corrector is not None:
            if self.ble_corrector.signal_fingerprints is not None:
                trajectory = self.ble_corrector.correct_initial_direction_with_fp(
                    trajectory
                )
            elif self.ble_corrector.transmitter_positions is not None:
                trajectory = (
                    self.ble_corrector.correct_initial_direction_with_transmitter_positions(
                        trajectory
                    )
                )

        elif self.map_matcher is not None:
            trajectory = self.map_matcher.correct_initial_direction()

        # if self.map_matcher is not None:
        #     trajectory = self.map_matcher.correct_unwalkable_points(trajectory)

        return trajectory

    @staticmethod
    def builder(pdr_estimator: PDREstimator) -> TrajectoryCorrectorsBuilder:
        return TrajectoryCorrectorsBuilder(pdr_estimator)


class TrajectoryCorrectorsBuilder:
    def __init__(self, pdr_estimator: PDREstimator) -> None:
        self.pdr_estimator = pdr_estimator
        self._floor_map: FloorMap | None = None
        self._gt_data: pd.DataFrame | None = None
        self._ble_realtime_scans: pd.DataFrame | None = None
        self._beacon_positions: pd.DataFrame | None
        self._ble_fingerprints: pd.DataFrame | None
        self._ble_config: dict = {"rssi_threshold": -75, "time_window": 5}

    def with_floor_map(self, floor_map: FloorMap) -> TrajectoryCorrectorsBuilder:
        self._floor_map = floor_map
        return self

    def with_ground_truth(
        self, gt_data: pd.DataFrame | None = None
    ) -> TrajectoryCorrectorsBuilder:
        self._gt_data = gt_data
        return self

    def with_wireless_signal(
        self,
        signal_realtime_scans: pd.DataFrame,
        signal_fingerprints: pd.DataFrame | None = None,
        transmitter_positions: pd.DataFrame | None = None,
        rssi_threshold: int | None = None,
        time_window: int | None = None,
    ) -> TrajectoryCorrectorsBuilder:
        self._ble_realtime_scans = signal_realtime_scans
        self._beacon_positions = transmitter_positions
        self._ble_fingerprints = signal_fingerprints

        # パラメータが指定された場合のみ更新
        if rssi_threshold is not None:
            self._ble_config["rssi_threshold"] = rssi_threshold
        if time_window is not None:
            self._ble_config["time_window"] = time_window

        return self

    def build(self) -> TrajectoryCorrector:
        drift_corrector = DriftCorrector({}, self.pdr_estimator, self._gt_data)
        map_matcher = (
            MapMatcher({}, self.pdr_estimator, self._floor_map)
            if self._floor_map is not None
            else None
        )
        ble_corrector = (
            WirelessSignalCorrector(
                signal_realtime_scans=self._ble_realtime_scans,
                transmitter_positions=self._beacon_positions,
                signal_fingerprints=self._ble_fingerprints,
                rssi_threshold=self._ble_config["rssi_threshold"],
                time_window=self._ble_config["time_window"],
            )
            if self._ble_realtime_scans is not None
            else None
        )

        return TrajectoryCorrector(
            pdr_estimator=self.pdr_estimator,
            drift_corrector=drift_corrector,
            map_matcher=map_matcher,
            ble_corrector=ble_corrector,
        )
