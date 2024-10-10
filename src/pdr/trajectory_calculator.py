import numpy as np
import pandas as pd

from src.const import ANGLE, COORDINATE_X, COORDINATE_Y, TIMESTAMP


class TrajectoryCalculator:
    def __init__(self, step_length: float = 0.5) -> None:
        self.step_length = step_length

    def calculate_trajectory(
        self,
        step_orientation: pd.DataFrame,
    ) -> pd.DataFrame:
        initial_point = {
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
                COORDINATE_X: x_moves.cumsum() + initial_point["x"],
                COORDINATE_Y: y_moves.cumsum() + initial_point["y"],
            },
        )
        # 空のエントリを除外
        new_trajectory = new_trajectory.dropna(how="all")

        # 連結
        return pd.concat([trajectory, new_trajectory])
