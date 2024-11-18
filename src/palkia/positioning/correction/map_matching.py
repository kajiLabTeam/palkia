from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from palkia.const import ANGLE, COORDINATE_X, COORDINATE_Y, TIMESTAMP
from palkia.positioning.pdr import PDREstimator
from palkia.utils.floor_map import FloorMap

Axis2D = Literal["x", "y"]


class MapMatcher:
    def __init__(
        self, config: dict[str, Any], pdr_estimator: PDREstimator, floor_map: FloorMap
    ) -> None:
        self.config = config
        self.floor_map = floor_map
        self.pdrEstimator = pdr_estimator

    def correct_initial_direction(self) -> pd.DataFrame:
        if self.pdrEstimator.enhanced_sensor_data.corrected_orrientation_df is None:
            msg = "Corrected orientation data is required for initial direction correction"
            raise ValueError(msg)

        step_times_orientations = self.pdrEstimator.estimate_step_times_orientations(
            self.pdrEstimator.enhanced_sensor_data.corrected_orrientation_df
        )
        # 初期方向の推定
        rotate_best_initial_direction = self.__find_best_initial_direction(
            step_times_orientations
        )

        corrected_angle_df = pd.DataFrame(
            {
                TIMESTAMP: step_times_orientations[TIMESTAMP],
                ANGLE: (step_times_orientations[ANGLE]) + rotate_best_initial_direction,
            }
        )

        return self.pdrEstimator.estimate_trajectory_from_orientation(
            corrected_angle_df
        )

    def _calculate_exist_counts(
        self,
        angle_df: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        def process(row) -> int:
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

    def __find_best_initial_direction(
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

        vertical_count = len(
            rotated_angle[
                (
                    (rotated_angle >= np.pi / 2 - 0.1)
                    & (rotated_angle <= np.pi / 2 + 0.1)
                )
                | (
                    (rotated_angle >= 3 * np.pi / 2 - 0.1)
                    & (rotated_angle <= 3 * np.pi / 2 + 0.1)
                )
            ],
        )

        horizontal_count = len(
            rotated_angle[
                ((rotated_angle <= 0.1) | (rotated_angle >= 2 * np.pi - 0.1))
                | ((rotated_angle >= np.pi - 0.1) & (rotated_angle <= np.pi + 0.1))
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
            results[results["exist_count"] == max_exist_count]
            .sort_values(by="horizontal_and_vertical_count", ascending=False)
            .iloc[0]
        )
        return optimal_result["angle"]

    # 既存の__init__等は省略

    def move_walkable_points(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """歩行不可能な点を歩行可能な点に移動する.

        Args:
        ----
            trajectory (pd.DataFrame): 補正する軌跡データ

        Returns:
        -------
            pd.DataFrame: 補正後の軌跡データ

        Raises:
        ------
            ValueError: 歩行可能な点が全く見つからない場合

        """
        if trajectory.empty:
            return trajectory

        corrected_trajectory = trajectory.copy()
        walkable_point_found = False

        for idx, row in trajectory.iterrows():
            print(row)
            if not self.floor_map.is_passable(row[COORDINATE_X], row[COORDINATE_Y]):
                nearest_point = self._get_nearest_walkable_point(
                    row[COORDINATE_X], row[COORDINATE_Y]
                )

                if nearest_point is None:
                    continue

                walkable_point_found = True
                delta_x = nearest_point[0] - row[COORDINATE_X]
                delta_y = nearest_point[1] - row[COORDINATE_Y]

                # 現在の点以降の全ての点を平行移動
                corrected_trajectory.loc[idx:, COORDINATE_X] += delta_x
                corrected_trajectory.loc[idx:, COORDINATE_Y] += delta_y

        if not walkable_point_found:
            raise ValueError("No walkable points could be found for the trajectory")

        return corrected_trajectory

    def _get_nearest_walkable_point(
        self, x: float, y: float
    ) -> tuple[float, float] | None:
        """最も近い歩行可能な点を見つける.

        Args:
        ----
            x (float): 現在のx座標
            y (float): 現在のy座標

        Returns:
        -------
            Optional[Tuple[float, float]]: 最も近い歩行可能な点。見つからない場合はNone

        """
        # BFS用のキュー初期化
        queue = deque([(x, y)])
        visited = {(x, y)}

        # スケールファクタ（メートルからピクセルへの変換）
        dx = self.floor_map.dx
        dy = self.floor_map.dy

        while queue:
            current_x, current_y = queue.popleft()

            # 現在の座標が歩行可能なら返す
            if self.floor_map.is_passable(current_x, current_y):
                return (current_x, current_y)

            # 隣接点を探索 (上下左右)
            for next_x, next_y in [
                (current_x + dx, current_y),
                (current_x - dx, current_y),
                (current_x, current_y + dy),
                (current_x, current_y - dy),
            ]:
                if (next_x, next_y) not in visited:
                    visited.add((next_x, next_y))
                    queue.append((next_x, next_y))

        return None
