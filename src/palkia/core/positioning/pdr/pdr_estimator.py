import pandas as pd

from palkia.config.column_name import TIMESTAMP
from palkia.core.sensor.data_preprocessor import match_data
from palkia.core.sensor.enhanced_sensor_data import EnhancedSensorData

from .orientation_estimator import OrientationEstimator
from .step_estimator import StepEstimator
from .trajectory_calculator import TrajectoryCalculator


class PDREstimator:
    def __init__(
        self,
        enhanced_sensor_data: EnhancedSensorData,
        step_estimator: StepEstimator,
        orientation_estimator: OrientationEstimator,
        trajectory_calculator: TrajectoryCalculator,
    ) -> None:
        self.enhanced_sensor_data = enhanced_sensor_data
        self.step_estimator = step_estimator
        self.orientation_estimator = orientation_estimator
        self.trajectory_calculator = trajectory_calculator

    def estimate_trajectory(
        self,
    ) -> pd.DataFrame:
        step_times = self.step_estimator.detect_step_times(
            self.enhanced_sensor_data.acc_df,
        )
        step_orientations = self.orientation_estimator.estimate_step_orientations(
            self.enhanced_sensor_data.gyro_df,
            pd.Series(step_times),
        )
        steps_lengths = self.step_estimator.estimate_steps(
            self.enhanced_sensor_data.acc_df, self.enhanced_sensor_data.gyro_df
        )
        # step_orientationsとsteps_lengthsを結合
        steps_data = step_orientations.merge(steps_lengths, on=TIMESTAMP)

        return self.trajectory_calculator.calculate_trajectory(steps_data)

    # orientationを使用して軌跡を推定する関数
    def estimate_trajectory_from_orientation(
        self,
        orientation_data: pd.DataFrame,
    ) -> pd.DataFrame:
        step_times = self.step_estimator.detect_step_times(
            self.enhanced_sensor_data.acc_df
        )
        step_lengths = self.step_estimator.estimate_steps_from_orientation(
            self.enhanced_sensor_data.acc_df, orientation_data
        )
        step_times_orientation = match_data(orientation_data, pd.Series(step_times))

        step_data = step_times_orientation.merge(step_lengths, on=TIMESTAMP)

        return self.trajectory_calculator.calculate_trajectory(step_data)

    def estimate_step_times_orientations(
        self,
        orientation_data: pd.DataFrame,
    ) -> pd.DataFrame:
        step_times = self.step_estimator.detect_step_times(
            self.enhanced_sensor_data.acc_df,
        )
        return match_data(orientation_data, pd.Series(step_times))
