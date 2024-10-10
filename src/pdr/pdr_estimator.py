import pandas as pd

from .orientation_estimator import OrientationEstimator
from .step_detector import StepDetector
from .trajectory_calculator import TrajectoryCalculator


class PDREstimator:
    def __init__(
        self,
        step_detector: StepDetector,
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
        steps_ts = self.step_detector.detect_steps_ts(acc_data)
        orientation = self.orientation_estimator.estimate_orientation(gyro_data)
        trajectory = self.trajectory_calculator.calculate_trajectory(
            steps_ts,
            orientation,
        )

        return trajectory
