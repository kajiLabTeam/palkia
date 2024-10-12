import pandas as pd

from src.palkia.const.column_name import TIMESTAMP

from .orientation_estimator import OrientationEstimator
from .step_estimator import StepEstimator
from .trajectory_calculator import TrajectoryCalculator


class PDREstimator:
    def __init__(
        self,
        step_estimator: StepEstimator,
        orientation_estimator: OrientationEstimator,
        trajectory_calculator: TrajectoryCalculator,
    ) -> None:
        self.step_estimator = step_estimator
        self.orientation_estimator = orientation_estimator
        self.trajectory_calculator = trajectory_calculator

    def estimate_trajectory(
        self,
        acc_data: pd.DataFrame,
        gyro_data: pd.DataFrame,
    ) -> pd.DataFrame:
        step_times = self.step_estimator.detect_step_times(acc_data)
        step_orientations = self.orientation_estimator.estimate_step_orientations(
            gyro_data,
            pd.Series(step_times),
        )
        steps_lengths = self.step_estimator.estimate_steps(acc_data, step_orientations)
        # step_orientationsとsteps_lengthsを結合
        steps_data = step_orientations.merge(steps_lengths, on=TIMESTAMP)

        return self.trajectory_calculator.calculate_trajectory(steps_data)
