from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.cluster import DBSCAN

from palkia.config import PRESSURE, TIMESTAMP
from palkia.core.positioning.floor_identification.floor_segments import FloorInfo

if TYPE_CHECKING:
    import pandas as pd

    from palkia.core.map.floor_map import FloorMap

# 標準大気圧の高度による変化:約12Pa/m(1気圧=1013.25hPa)
PRESSURE_CHANGE_PER_METER = 0.012  # hPa/m


class FloorIdentifier:
    """Identifies floor levels based on barometer data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize FloorIdentifier.

        Args:
        ----
            config: Configuration dictionary containing:
                - pressure_threshold (float): Maximum allowed pressure variation within a stable interval.
                  Default is 0.02 hPa, corresponding to about 1.7m height difference.
                - stable_duration (float): Minimum time duration (in seconds) required for a stable
                  pressure reading. Default is 4 seconds to ensure reliable floor detection.
                - gap_threshold (float): Maximum allowed time gap (in seconds) between consecutive
                  measurements to be considered part of the same interval. Default is 1.0 second.
                - floor_height (float): Typical floor height in meters. Default is 3.0 meters.
                - base_pressure (float): Reference pressure at sea level. Default is 1013.25 hPa.
                - dbscan_min_samples (int): Minimum number of samples to form a core cluster.
                  Default is 1 to accommodate limited data scenarios.

        """
        self.config = config or {}
        self.pressure_threshold = self.config.get("pressure_threshold", 0.02)
        self.stable_duration = self.config.get("stable_duration", 4)
        self.floor_height_meters = self.config.get("floor_height", 3.0)
        self.base_pressure = self.config.get("base_pressure", 1013.25)
        # 階高の半分を許容
        self.dbscan_eps = PRESSURE_CHANGE_PER_METER * self.floor_height_meters * 0.5

        self.dbscan_min_samples = self.config.get("dbscan_min_samples", 1)

    def identify_floors(
        self,
        baro_data: pd.DataFrame,
        trajectory: pd.DataFrame,
        floor_maps: dict[int, FloorMap] | None = None,
    ) -> dict[int, FloorInfo]:
        """階層を識別し、各階層の情報を生成する.

        Args:
        ----
            baro_data: 気圧センサーデータ
            trajectory: 軌跡データ
            floor_maps: 各階のフロアマップ(オプション)

        Returns:
        -------
            各階層の情報を含む辞書

        """
        # 安定区間の検出
        stable_intervals = self._find_stable_intervals(baro_data)

        # 階層ごとの気圧レベルを特定
        pressure_levels = self._group_pressure_levels(baro_data, stable_intervals)

        # floor_segmentsオブジェクトの生成
        floor_segments = self._create_floor_segments(pressure_levels, trajectory)

        # FloorMapの関連付け
        if floor_maps is not None:
            for floor_num, info in floor_segments.items():
                if floor_num in floor_maps:
                    info.floor_map = floor_maps[floor_num]

        return floor_segments

    def _preprocess_pressure_data(self, baro_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess pressure data with noise removal and smoothing.

        Args:
        ----
            baro_data: Raw barometer data.

        Returns:
        -------
            Processed barometer data.

        """
        processed_data = baro_data.copy()

        # Remove outliers using 3-sigma method
        mean_pressure = processed_data[PRESSURE].mean()
        std_pressure = processed_data[PRESSURE].std()
        processed_data.loc[
            (processed_data[PRESSURE] - mean_pressure).abs() > 3 * std_pressure,
            PRESSURE,
        ] = np.nan
        processed_data[PRESSURE] = processed_data[PRESSURE].interpolate(method="linear")

        # Apply smoothing
        window_size = self.config.get("smoothing_window", 5)
        processed_data[PRESSURE] = (
            processed_data[PRESSURE]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        return processed_data

    def _find_stable_intervals(
        self, baro_data: pd.DataFrame
    ) -> list[tuple[float, float]]:
        """Find time intervals with stable pressure readings.

        Args:
        ----
            baro_data: Preprocessed barometer data.

        Returns:
        -------
            List of (start_time, end_time) tuples for stable intervals.

        """
        stable_intervals = []
        window_size = int(self.stable_duration)
        start_idx = None

        for i in range(len(baro_data) - window_size):
            window = baro_data.iloc[i : i + window_size]
            pressure_range = window[PRESSURE].max() - window[PRESSURE].min()

            if pressure_range <= self.pressure_threshold:
                if start_idx is None:
                    start_idx = i

            elif start_idx is not None:
                stable_intervals.append(
                    (baro_data[TIMESTAMP].iloc[start_idx], baro_data[TIMESTAMP].iloc[i])
                )
                start_idx = None

        # 最後の安定区間のチェック
        if start_idx is not None:
            # 最後のウィンドウをチェック
            last_window = baro_data.iloc[start_idx:]
            if len(last_window) >= window_size:
                pressure_range = (
                    last_window[PRESSURE].max() - last_window[PRESSURE].min()
                )
                if pressure_range <= self.pressure_threshold:
                    stable_intervals.append(
                        (
                            baro_data[TIMESTAMP].iloc[start_idx],
                            baro_data[TIMESTAMP].iloc[-1],  # 最後のタイムスタンプ
                        )
                    )

        return stable_intervals

    def _group_pressure_levels(
        self, baro_data: pd.DataFrame, stable_intervals: list[tuple[float, float]]
    ) -> dict[int, float]:
        """Group stable intervals into distinct floor levels using DBSCAN.

        Args:
        ----
            baro_data: Preprocessed barometer data.
            stable_intervals: List of stable time intervals.

        Returns:
        -------
            Dictionary mapping floor numbers to representative pressure values.

        """
        # 安定区間から平均気圧値を抽出
        pressure_values = []
        for start, end in stable_intervals:
            interval_data = baro_data[
                (baro_data[TIMESTAMP] >= start) & (baro_data[TIMESTAMP] <= end)
            ]
            pressure_values.append(interval_data[PRESSURE].mean())

        if not pressure_values:
            return {}

        # DBSCANでクラスタリング
        X = np.array(pressure_values).reshape(-1, 1)
        clustering = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit(X)

        # クラスタごとの平均気圧を計算
        floor_pressures = {}
        for label in set(clustering.labels_):
            if label != -1:  # ノイズを除外
                mask = clustering.labels_ == label
                floor_pressures[label] = np.mean(X[mask])

        return self._normalize_floor_numbers(floor_pressures)


    def _normalize_floor_numbers(
        self, floor_pressures: dict[int, float]
    ) -> dict[int, float]:
        """Normalize floor numbers starting from the lowest pressure (highest floor)."""
        pressures = sorted(
            floor_pressures.values(), reverse=True
        )  # 圧力の高い順(低層階から)
        return dict(enumerate(pressures))

    def _create_floor_segments(
        self, floor_pressures: dict[int, float], trajectory: pd.DataFrame
    ) -> dict[int, FloorInfo]:
        """Create FloorInfo objects for each identified floor.

        Args:
        ----
            floor_pressures: Dictionary mapping floor numbers to pressures.
            trajectory: Complete trajectory data.

        Returns:
        -------
            Dictionary mapping floor numbers to FloorInfo objects.

        """
        floor_segments: dict[int, FloorInfo] = {}

        for floor_num, pressure in floor_pressures.items():
            pressure_range = (
                pressure - self.pressure_threshold,
                pressure + self.pressure_threshold,
            )

            floor_trajectory = self._extract_floor_trajectory(
                trajectory, pressure_range
            )
            time_intervals = self._get_continuous_time_intervals(floor_trajectory)

            floor_segments[floor_num] = FloorInfo(
                floor_number=floor_num,
                pressure_range=pressure_range,
                time_intervals=time_intervals,
                trajectory=floor_trajectory,
            )

        return floor_segments

    def _extract_floor_trajectory(
        self, trajectory: pd.DataFrame, pressure_range: tuple[float, float]
    ) -> pd.DataFrame:
        """Extract trajectory data for a specific pressure range.

        Args:
        ----
            trajectory: Complete trajectory data.
            pressure_range: Tuple of (min_pressure, max_pressure).

        Returns:
        -------
            Trajectory data within the specified pressure range.

        """
        min_pressure, max_pressure = pressure_range
        return trajectory[
            (trajectory[PRESSURE] >= min_pressure)
            & (trajectory[PRESSURE] <= max_pressure)
        ].copy()

    def _get_continuous_time_intervals(
        self, trajectory: pd.DataFrame
    ) -> list[tuple[float, float]]:
        """Find continuous time intervals in trajectory data.

        Args:
        ----
            trajectory: Floor-specific trajectory data.

        Returns:
        -------
            List of (start_time, end_time) tuples.

        """
        if trajectory.empty:
            return []

        intervals = []
        start_time = trajectory[TIMESTAMP].iloc[0]
        prev_time = start_time

        for _, row in trajectory.iloc[1:].iterrows():
            if row[TIMESTAMP] - prev_time > self.config.get("gap_threshold", 1.0):
                intervals.append((start_time, prev_time))
                start_time = row[TIMESTAMP]
            prev_time = row[TIMESTAMP]

        intervals.append((start_time, prev_time))
        return intervals
