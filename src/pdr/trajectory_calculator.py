from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.const import ANGLE, COORDINATE_X, COORDINATE_Y, TIMESTAMP


class TrajectoryCalculator:
    def __init__(
        self,
        step_length: float = 0.5,
        initial_point: dict[Literal["x", "y"], float] | None = None,
    ) -> None:
        self.step_length = step_length
        self.initial_point = initial_point

    def calculate_trajectory(
        self,
        step_orientation: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.initial_point is None:
            self.initial_point = {
                "x": 0,
                "y": 0,
            }

        x_moves = self.step_length * np.cos(step_orientation[ANGLE])
        y_moves = self.step_length * np.sin(step_orientation[ANGLE])

        initial_point_df = pd.DataFrame(
            {
                TIMESTAMP: [step_orientation[TIMESTAMP][0]],
                COORDINATE_X: [self.initial_point["x"]],
                COORDINATE_Y: [self.initial_point["y"]],
            },
        )

        trajectory = pd.concat(
            [
                initial_point_df,
                pd.DataFrame(
                    {
                        TIMESTAMP: step_orientation[TIMESTAMP],
                        COORDINATE_X: self.initial_point["x"] + x_moves.cumsum(),
                        COORDINATE_Y: self.initial_point["y"] + y_moves.cumsum(),
                    },
                ),
            ],
        )

        # 空のエントリを除外
        return trajectory.dropna(how="all")
