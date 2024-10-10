import numpy as np
import pandas as pd

from src.const import ANGLE, COORDINATE_X, COORDINATE_Y, TIMESTAMP


class TrajectoryCalculator:
    def __init__(self, step_length: float = 0.5) -> None:
        self.step_length = step_length

    @staticmethod
    def __match_data(something_df: pd.DataFrame, peek_t: pd.Series) -> pd.DataFrame:
        matched_df = pd.DataFrame()
        for t in peek_t:
            matched_row = something_df[np.isclose(something_df["ts"], t, atol=0.005)]
            matched_df = pd.concat([matched_df, matched_row])
        return matched_df.reset_index(drop=True)

    def calculate_trajectory(
        self,
        steps_ts: np.ndarray,
        orientation: pd.DataFrame,
    ) -> pd.DataFrame:
        initial_point = {
            "x": 0,
            "y": 0,
        }

        trajectory = pd.DataFrame(columns=[TIMESTAMP, "x", "y"])

        peek_orientation = TrajectoryCalculator.__match_data(
            orientation,
            pd.Series(steps_ts),
        )

        x_moves = self.step_length * np.cos(peek_orientation[ANGLE])
        y_moves = self.step_length * np.sin(peek_orientation[ANGLE])

        return pd.concat(
            [
                trajectory,
                pd.DataFrame(
                    {
                        TIMESTAMP: steps_ts,
                        COORDINATE_X: x_moves.cumsum() + initial_point["x"],
                        COORDINATE_Y: y_moves.cumsum() + initial_point["y"],
                    },
                ),
            ],
        )
