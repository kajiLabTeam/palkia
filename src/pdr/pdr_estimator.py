import pandas as pd

from .orientation_estimator import OrientationEstimator
from .step_estimator import StepEstimator
from .trajectory_calculator import TrajectoryCalculator


class PDREstimator:
    def __init__(
        self,
        step_detector: StepEstimator,
        orientation_estimator: OrientationEstimator,
        trajectory_calculator: TrajectoryCalculator,
    ) -> None:
        self.step_detector = step_detector
        self.orientation_estimator = orientation_estimator
        self.trajectory_calculator = trajectory_calculator

    def estimate_trajectory(
        self,
        acc_data: pd.DataFrame,
        gyro_data: pd.DataFrame,
    ) -> pd.DataFrame:
        step_times = self.step_detector.detect_step_times(acc_data)
        step_orientation = self.orientation_estimator.estimate_step_orientation(
            gyro_data,
            pd.Series(step_times),
        )
        return self.trajectory_calculator.calculate_trajectory(
            step_orientation,
        )
