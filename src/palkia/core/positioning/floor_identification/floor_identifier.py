from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from palkia.config import PRESSURE, TIMESTAMP
from palkia.core.positioning.floor_identification.floor_info import FloorInfo

if TYPE_CHECKING:
    import pandas as pd

    from palkia.utils.floor_map import FloorMap


class FloorIdentifier:
    """Identifies floor levels based on barometer data."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize FloorIdentifier.

        Args:
        ----
            config: Configuration dictionary for floor identification parameters.

        """
        self.config = config or {}
        self.pressure_threshold = self.config.get("pressure_threshold", 0.02)
        self.stable_duration = self.config.get("stable_duration", 4)
        self.floor_height_meters = self.config.get("floor_height", 3.0)
        self.base_pressure = self.config.get("base_pressure", 1013.25)

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

        # floor_infoオブジェクトの生成
        floor_info = self._create_floor_info(pressure_levels, trajectory)

        # FloorMapの関連付け
        if floor_maps is not None:
            for floor_num, info in floor_info.items():
                if floor_num in floor_maps:
                    info.floor_map = floor_maps[floor_num]

        return floor_info

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
        """Group stable intervals into distinct floor levels.

        Args:
        ----
            baro_data: Preprocessed barometer data.
            stable_intervals: List of stable time intervals.

        Returns:
        -------
            Dictionary mapping floor numbers to representative pressure values.

        """
        pressure_values = []

        for start, end in stable_intervals:
            interval_data = baro_data[
                (baro_data[TIMESTAMP] >= start) & (baro_data[TIMESTAMP] <= end)
            ]
            pressure_values.append(interval_data[PRESSURE].mean())

        # Group similar pressure values
        pressure_values = np.array(pressure_values)
        floor_pressures = {}
        assigned = np.zeros_like(pressure_values, dtype=bool)

        floor_num = 0
        while not assigned.all():
            unassigned_idx = np.where(~assigned)[0][0]
            current_pressure = pressure_values[unassigned_idx]

            # Find all pressures within threshold
            similar_pressures = (
                np.abs(pressure_values - current_pressure) <= self.pressure_threshold
            )
            assigned[similar_pressures] = True

            floor_pressures[floor_num] = np.mean(pressure_values[similar_pressures])
            floor_num += 1

        return self._normalize_floor_numbers(floor_pressures)

    def _normalize_floor_numbers(
        self, floor_pressures: dict[int, float]
    ) -> dict[int, float]:
        """Normalize floor numbers to use 1 as ground floor.

        Args:
        ----
            floor_pressures: Dictionary of temporary floor numbers and pressures.

        Returns:
        -------
            Dictionary with normalized floor numbers.

        """
        pressures = np.array(list(floor_pressures.values()))
        floor_numbers = np.argsort(pressures)[::-1] - len(pressures) // 2
        return {
            int(floor_num): pressure
            for floor_num, pressure in zip(floor_numbers, pressures)
        }

    def _create_floor_info(
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
        floor_info: dict[int, FloorInfo] = {}

        for floor_num, pressure in floor_pressures.items():
            pressure_range = (
                pressure - self.pressure_threshold,
                pressure + self.pressure_threshold,
            )

            floor_trajectory = self._extract_floor_trajectory(
                trajectory, pressure_range
            )
            time_intervals = self._get_continuous_time_intervals(floor_trajectory)

            floor_info[floor_num] = FloorInfo(
                floor_number=floor_num,
                pressure_range=pressure_range,
                time_intervals=time_intervals,
                trajectory=floor_trajectory,
            )

        return floor_info

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
