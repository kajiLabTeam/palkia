from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from palkia.config.column_name import COORDINATE_X, COORDINATE_Y
from palkia.config.path import BEACON_LIST_PATH


@dataclass
class Point2D:
    x: float
    y: float


class BLECorrector:
    def __init__(
        self,
        beacon_positions: pd.DataFrame | None = None,
        rssi_threshold: int = -80,
        time_window: int = 5,
    ) -> None:
        self.beacon_positions = (
            pd.read_csv(BEACON_LIST_PATH)
            if beacon_positions is None
            else beacon_positions
        )
        self.rssi_threshold = rssi_threshold
        self.time_window = time_window

    def correct_initial_direction(
        self, trajectory: pd.DataFrame, ble_data: pd.DataFrame
    ) -> pd.DataFrame:
        """BLEデータを使用して軌跡を補正する.

        Args:
            trajectory: 補正する軌跡
            ble_data: BLEスキャンデータ

        Returns:
            BLE補正された軌跡

        """
        strong_ble_scans = self._filter_strong_blescans(ble_data)
        strong_ble_merged = strong_ble_scans.merge(
            self.beacon_positions, on="bdaddress", how="left"
        ).rename(columns={"x": "ble_x", "y": "ble_y"})

        initial_point = Point2D(
            x=trajectory[COORDINATE_X].iloc[0], y=trajectory[COORDINATE_Y].iloc[0]
        )

        return self._optimize_trajectory_rotation(
            trajectory.copy(), strong_ble_merged, initial_point
        )

    def _rotate_trajectory(
        self, df: pd.DataFrame, angle: float, initial_point: Point2D
    ) -> pd.DataFrame:
        x_displacement = df[COORDINATE_X] - initial_point.x
        y_displacement = df[COORDINATE_Y] - initial_point.y

        rotated_x = (
            x_displacement * np.cos(angle)
            - y_displacement * np.sin(angle)
            + initial_point.x
        )
        rotated_y = (
            x_displacement * np.sin(angle)
            + y_displacement * np.cos(angle)
            + initial_point.y
        )

        return pd.DataFrame(
            {"ts": df.ts, COORDINATE_X: rotated_x, COORDINATE_Y: rotated_y}
        )

    def _calculate_total_distance(
        self, trajectory: pd.DataFrame, ble_data: pd.DataFrame
    ) -> float:
        merged = pd.merge_asof(
            trajectory.sort_values("ts"),
            ble_data.sort_values("ts"),
            on="ts",
            direction="nearest",
        )

        return np.sqrt(
            (merged[COORDINATE_X] - merged["ble_x"]) ** 2
            + (merged[COORDINATE_Y] - merged["ble_y"]) ** 2
        ).sum()

    def _optimize_trajectory_rotation(
        self,
        trajectory: pd.DataFrame,
        ble_data: pd.DataFrame,
        initial_point: Point2D,
    ) -> pd.DataFrame:
        angles = np.arange(0, 2 * np.pi, 0.01)
        optimal_angle = min(
            angles,
            key=lambda angle: self._calculate_total_distance(
                self._rotate_trajectory(trajectory, angle, initial_point), ble_data
            ),
        )

        return self._rotate_trajectory(trajectory, optimal_angle, initial_point)

    def _filter_strong_blescans(
        self, ble_data: pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        """強いRSSI値のBLEスキャンのみをフィルタリングする."""
        return ble_data[ble_data["rssi"] > self.rssi_threshold].copy()
