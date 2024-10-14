from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from palkia.positioning.pdr import PDREstimator
from palkia.utils.floor_map import FloorMap


class MapMatcher:
    def __init__(
        self, config: Dict[str, Any], pdr_estimator: PDREstimator, floor_map: FloorMap
    ) -> None:
        self.config = config
        self.floor_map = floor_map
        self.pdrEstimator = pdr_estimator

    def _find_best_alignment_angle(
        self,
        acc_df: pd.DataFrame,
        angle_df: pd.DataFrame,
        ground_truth_first_point: dict[Axis2D, float],
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
            ground_truth_first_point,
        )
        return self._get_optimal_angle(df_results)

    def _calculate_exist_counts(
        self,
        acc_df: pd.DataFrame,
        angle_df: pd.DataFrame,
        results: pd.DataFrame,
        ground_truth_first_point: dict[Axis2D, float],
    ) -> pd.DataFrame:
        for i, row in results.head(20).iterrows():
            rotated_displacement = PDREstimator.estimate_trajectory_from_orientation(
                angle_df.ts,
                (angle_df["x"] + row["angle"]),
                0.5,
                {
                    "x": ground_truth_first_point["x"],
                    "y": ground_truth_first_point["y"],
                },
            )

            exist_count = 0
            for _, displacement_row in rotated_displacement.iterrows():
                if _is_passable(
                    edit_map_dict,
                    floor_name,
                    displacement_row["x_displacement"],
                    displacement_row["y_displacement"],
                    dx,
                    dy,
                ):
                    exist_count += 1

            results.at[i, "exist_count"] = exist_count

        return results

    def _calculate_horizontal_and_vertical_counts(
        self,
        angle_df: pd.DataFrame,
        rotate_angle: float,
    ) -> dict[str, int | float]:
        rotated_angle = (angle_df["x"] + rotate_angle) % (2 * np.pi)

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

    def _get_optimal_angle(self, results: pd.DataFrame) -> float:
        max_exist_count = results["exist_count"].max()
        optimal_result = (
            results[results["exist_count"] == max_exist_count]
            .sort_values(by="horizontal_and_vertical_count", ascending=False)
            .iloc[0]
        )
        return optimal_result["angle"]
