from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from palkia.config import ANGLE, COORDINATE_X, COORDINATE_Y, TIMESTAMP

if TYPE_CHECKING:
    from palkia.core.map.floor_map import FloorMap
    from palkia.core.positioning.pdr import PDREstimator


class MapMatchCorrector:
    def __init__(
        self, config: dict[str, Any], pdr_estimator: PDREstimator, floor_map: FloorMap
    ) -> None:
        self.config = config
        self.floor_map = floor_map
        self.pdrEstimator = pdr_estimator

    def correct_initial_direction(self) -> pd.DataFrame:
        step_times_orientations = self.pdrEstimator.estimate_step_times_orientations(
            self.pdrEstimator.enhanced_sensor_data.get_corrected_orrientation_df()
        )

        # 初期方向の推定
        rotate_best_initial_direction = self._find_best_initial_direction(
            step_times_orientations
        )

        corrected_angle_df = pd.DataFrame(
            {
                TIMESTAMP: step_times_orientations[TIMESTAMP],
                ANGLE: (step_times_orientations[ANGLE]) + rotate_best_initial_direction,
            }
        )

        self.pdrEstimator.enhanced_sensor_data.update_corrected_orrientation_df(
            corrected_angle_df
        )

        return self.pdrEstimator.estimate_trajectory_from_orientation(
            corrected_angle_df
        )

    def correct_unwalkable_points(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:
        corrected_trajectory = trajectory.copy().reset_index(drop=True)

        for index, _ in enumerate(trajectory.iterrows()):
            nearest_row = corrected_trajectory.iloc[index]
            if not self.floor_map.is_passable(
                nearest_row[COORDINATE_X],
                nearest_row[COORDINATE_Y],
            ):
                corrected_trajectory = self._apply_trajectory_correction(
                    corrected_trajectory,
                    index,
                    nearest_row,
                )

        return corrected_trajectory

    def _calculate_exist_counts(
        self,
        angle_df: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        def process(row: pd.DataFrame) -> int:
            rotated_displacement = (
                self.pdrEstimator.estimate_trajectory_from_orientation(
                    pd.DataFrame(
                        {
                            TIMESTAMP: angle_df[TIMESTAMP],
                            ANGLE: (angle_df[ANGLE] + row["angle"]),
                        }
                    )
                )
            )

            return rotated_displacement.apply(
                lambda x: self.floor_map.is_passable(x[COORDINATE_X], x[COORDINATE_Y]),
                axis=1,
            ).sum()

        results.loc[:19, "exist_count"] = results.head(20).apply(process, axis=1)

        return results

    def _find_best_initial_direction(
        self,
        angle_df: pd.DataFrame,
    ) -> float:
        angle_range = np.arange(0, 2 * np.pi, 0.01)
        results = [
            self._calculate_horizontal_and_vertical_counts(angle_df, rotate_angle)
            for rotate_angle in angle_range
        ]
        df_results = pd.DataFrame(results).sort_values(
            by="horizontal_and_vertical_count",
            ascending=False,
        )
        df_results = df_results.reset_index(drop=True)

        df_results = self._calculate_exist_counts(
            angle_df,
            df_results,
        )
        return self._get_optimal_angle(df_results)

    def _calculate_horizontal_and_vertical_counts(
        self,
        angle_df: pd.DataFrame,
        rotate_angle: float,
    ) -> dict[str, int | float]:
        rotated_angle = (angle_df[ANGLE] + rotate_angle) % (2 * np.pi)
        rad_range_threshold = 0.1

        vertical_count = len(
            rotated_angle[
                (
                    (rotated_angle >= np.pi / 2 - rad_range_threshold)
                    & (rotated_angle <= np.pi / 2 + rad_range_threshold)
                )
                | (
                    (rotated_angle >= 3 * np.pi / 2 - rad_range_threshold)
                    & (rotated_angle <= 3 * np.pi / 2 + rad_range_threshold)
                )
            ],
        )

        horizontal_count = len(
            rotated_angle[
                (
                    (rotated_angle <= rad_range_threshold)
                    | (rotated_angle >= 2 * np.pi - rad_range_threshold)
                )
                | (
                    (rotated_angle >= np.pi - rad_range_threshold)
                    & (rotated_angle <= np.pi + rad_range_threshold)
                )
            ],
        )

        return {
            "angle": rotate_angle,
            "horizontal_and_vertical_count": horizontal_count + vertical_count,
        }

    # exist_countを優先するようにソートする
    def _get_optimal_angle(self, results: pd.DataFrame) -> float:
        max_exist_count = results["exist_count"].max()
        optimal_result = (
            results[results["exist_count"] == max_exist_count]  # type: ignore
            .sort_values(by=("horizontal_and_vertical_count"), ascending=False)
            .iloc[0]
        )
        return optimal_result["angle"]

    def _apply_trajectory_correction(
        self,
        corrected_displacement_df: pd.DataFrame,
        index: int,
        nearest_row: pd.Series,
    ) -> pd.DataFrame:
        before_of_correction_point = {
            "x": nearest_row[COORDINATE_X],
            "y": nearest_row[COORDINATE_Y],
        }

        corrected_point = self._find_nearest_passable_point_dijkstra(
            float(nearest_row[COORDINATE_X]),
            float(nearest_row[COORDINATE_Y]),
        )

        if corrected_point is None:
            return corrected_displacement_df

        after_of_correction_point = {
            "x": corrected_point[0],
            "y": corrected_point[1],
        }

        delta_x = after_of_correction_point["x"] - before_of_correction_point["x"]
        delta_y = after_of_correction_point["y"] - before_of_correction_point["y"]

        corrected_displacement_df.loc[index:, [COORDINATE_X, COORDINATE_Y]] += [
            delta_x,
            delta_y,
        ]

        return corrected_displacement_df

    def _find_nearest_passable_point_dijkstra(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float] | None:
        # グリッド変換時のイプシロン値を追加
        epsilon = 1e-9
        start_row = int((x + epsilon) / self.floor_map.dx)
        start_col = int((y + epsilon) / self.floor_map.dy)

        if not self._is_in_bounds(start_row, start_col):
            return None

        queue = [(0, start_row, start_col)]
        visited = {(start_row, start_col)}

        directions = [
            ((-1, 0), 1),
            ((1, 0), 1),
            ((0, -1), 1),
            ((0, 1), 1),
            ((-1, -1), np.sqrt(2)),
            ((-1, 1), np.sqrt(2)),
            ((1, -1), np.sqrt(2)),
            ((1, 1), np.sqrt(2)),
        ]

        while queue:
            cost, row, col = heapq.heappop(queue)

            if self.floor_map.is_passable(
                row * self.floor_map.dx, col * self.floor_map.dy
            ):
                return row * self.floor_map.dx, col * self.floor_map.dy

            for (dr, dc), weight in directions:
                next_row, next_col = row + dr, col + dc
                if (
                    self._is_in_bounds(next_row, next_col)
                    and (next_row, next_col) not in visited
                ):
                    heapq.heappush(queue, (cost + weight, next_row, next_col))
                    visited.add((next_row, next_col))

        return None

    def _is_in_bounds(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.floor_map.floor_map_data.shape[0]
            and 0 <= col < self.floor_map.floor_map_data.shape[1]
        )
