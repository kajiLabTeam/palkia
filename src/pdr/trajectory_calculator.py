from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.const import ANGLE, COORDINATE_X, COORDINATE_Y, STEP_LENGTH, TIMESTAMP


class TrajectoryCalculator:
    def __init__(
        self,
        initial_point: dict[Literal["x", "y"], float] | None = None,
    ) -> None:
        self.initial_point = initial_point or {"x": 0, "y": 0}

    def calculate_trajectory(self, steps_data: pd.DataFrame) -> pd.DataFrame:
        x_moves = steps_data[STEP_LENGTH] * np.cos(steps_data[ANGLE])
        y_moves = steps_data[STEP_LENGTH] * np.sin(steps_data[ANGLE])

        initial_point_df = pd.DataFrame(
            {
                TIMESTAMP: [steps_data[TIMESTAMP][0]],
                COORDINATE_X: [self.initial_point["x"]],
                COORDINATE_Y: [self.initial_point["y"]],
            },
        )

        trajectory = pd.concat(
            [
                initial_point_df,
                pd.DataFrame(
                    {
                        TIMESTAMP: steps_data[TIMESTAMP],
                        COORDINATE_X: self.initial_point["x"] + x_moves.cumsum(),
                        COORDINATE_Y: self.initial_point["y"] + y_moves.cumsum(),
                    },
                ),
            ],
        )

        return trajectory.dropna(how="all")
