from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from palkia.const.column_name import (
    ANGLE,
    COORDINATE_X,
    COORDINATE_Y,
    GYRO_X,
    TIMESTAMP,
)

if TYPE_CHECKING:
    from palkia.positioning.pdr import PDREstimator


class DriftCorrector:
    def __init__(
        self, config: dict[str, Any], pdr_estimator: PDREstimator, gt_data: pd.DataFrame
    ) -> None:
        self.pdr_estimator = pdr_estimator
        self.gt_data = gt_data
        self.drift_search_range = config.get("drift_search_range", (-0.05, 0.05))
        self.drift_search_step = config.get("drift_search_step", 0.001)

    def correct(self, gyro_df: pd.DataFrame) -> pd.DataFrame:
        angle_df = self._convert_gyro_to_angle(gyro_df)
        optimal_drift = self._search_optimal_drift_from_angle(angle_df)
        corrected_angle_df = self._apply_drift_correction(angle_df, optimal_drift)
        return self.pdr_estimator.estimate_trajectory_from_orientation(
            corrected_angle_df
        )

    @staticmethod
    def _convert_gyro_to_angle(gyro_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                TIMESTAMP: gyro_df[TIMESTAMP],
                ANGLE: gyro_df[GYRO_X].cumsum() * 0.01,  # Assuming 100Hz sampling rate
            }
        )

    @staticmethod
    def _apply_drift_correction(angle_df: pd.DataFrame, drift: float) -> pd.DataFrame:
        corrected_angle = angle_df.copy()
        base_time = corrected_angle[TIMESTAMP].iloc[0]
        elapsed_time = corrected_angle[TIMESTAMP] - base_time
        corrected_angle[ANGLE] -= drift * elapsed_time
        return corrected_angle

    @staticmethod
    def _compute_euclidean_distance(df: pd.DataFrame, gt: pd.Series) -> float:
        last_row = df.iloc[-1]
        return np.sqrt(
            (last_row[COORDINATE_X] - gt.x) ** 2 + (last_row[COORDINATE_Y] - gt.y) ** 2
        )

    def _search_optimal_drift_from_angle(self, angle_df: pd.DataFrame) -> float:
        start, end = self.drift_search_range
        drift_range = np.arange(start, end, self.drift_search_step)

        def evaluate_drift(drift: float) -> float:
            adjusted_angle = self._apply_drift_correction(angle_df, drift)
            displacement_df = self.pdr_estimator.estimate_trajectory_from_orientation(
                adjusted_angle
            )
            return self._compute_euclidean_distance(
                displacement_df, self.gt_data.iloc[1]
            )

        drift_and_distance = [
            (drift, evaluate_drift(drift)) for drift in drift_range if abs(drift) < 0.01
        ]
        return min(drift_and_distance, key=lambda x: x[1])[0]
