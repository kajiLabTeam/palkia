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

        trajectory = pd.DataFrame(columns=[TIMESTAMP, "x", "y"])
        x_moves = self.step_length * np.cos(step_orientation[ANGLE])
        y_moves = self.step_length * np.sin(step_orientation[ANGLE])
        # 新しいデータフレームを作成
        new_trajectory = pd.DataFrame(
            {
                TIMESTAMP: step_orientation[TIMESTAMP],
                COORDINATE_X: x_moves.cumsum() + self.initial_point["x"],
                COORDINATE_Y: y_moves.cumsum() + self.initial_point["y"],
            },
        )
        # 空のエントリを除外
        new_trajectory = new_trajectory.dropna(how="all")

        # 連結
        return pd.concat([trajectory, new_trajectory])
