from __future__ import annotations

from typing import Any, Dict, Literal

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
        rotate_best_initial_direction = self.find_best_initial_direction(
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

    def find_best_initial_direction(
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
