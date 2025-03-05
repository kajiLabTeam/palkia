import pandas as pd

from palkia.config import (
    FLOOR_MAP_PATH,
    POS_X,
    POS_Y,
    STEP_LENGTH_MODEL_PATH,
    TIMESTAMP,
)
from palkia.core.map.floor_map import FloorMap
from palkia.core.positioning.correction import DriftCorrector, MapMatchCorrector
from palkia.core.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.core.positioning.pdr.pdr_estimator import PDREstimator
from palkia.core.positioning.pdr.step_estimator import StepEstimator
from palkia.core.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.core.sensor_processing.data_loader import load_sensor_data_from_log
from palkia.core.sensor_processing.enhanced_sensor_data import EnhancedSensorData
from palkia.core.visualization import plot_trajectory


def main() -> None:
    acc_data = pd.read_csv("../data/raw/other/ThreeDimensional/Accelerometer.csv")
    gyro_data = pd.read_csv("../data/raw/other/ThreeDimensional/Gyroscope.csv")

    # センサーデータの可視化
    gt_data = pd.DataFrame(
        {
            TIMESTAMP: [0, 200],
            POS_X: [15, 15],
            POS_Y: [19, 19],
        },
    )

    # PDR推定器の初期化
    pdr_estimator = PDREstimator(
        EnhancedSensorData(
            acc_data,
            gyro_data,
        ),
        StepEstimator(
            step_length_model_path=STEP_LENGTH_MODEL_PATH,
        ),
        OrientationEstimator(),
        TrajectoryCalculator(
            flip_vertical=True,
            initial_point={
                "x": gt_data.loc[0, POS_X],
                "y": gt_data.loc[0, POS_Y],
            },
        ),
    )

    # 軌跡の推定
    trajectory = pdr_estimator.estimate_trajectory()
    floor_name = "floor_5"

    floor_map = FloorMap(
        floor_name=floor_name,
        floor_map_path=FLOOR_MAP_PATH.format(floor_name, "png"),
        dx=0.01,
        dy=0.01,
    )

    plot_trajectory(trajectory, floor_map=floor_map)

    correct_drift_trajectory = DriftCorrector(
        config={}, pdr_estimator=pdr_estimator, gt_data=gt_data
    ).correct_drift()

    plot_trajectory(correct_drift_trajectory, floor_map=floor_map)

    correct_map_matching_trajectory = MapMatchCorrector(
        config={}, pdr_estimator=pdr_estimator, floor_map=floor_map
    ).correct_initial_direction()

    plot_trajectory(correct_map_matching_trajectory, floor_map=floor_map)



if __name__ == "__main__":
    main()
