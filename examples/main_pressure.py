import pandas as pd

from palkia.config import (
    FLOOR_MAP_PATH,
    POS_X,
    POS_Y,
    STEP_LENGTH_MODEL_PATH,
    TIMESTAMP,
)
from palkia.core.map.floor_map import FloorMap
from palkia.core.positioning.correction import DriftCorrector, MapMatcher
from palkia.core.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.core.positioning.pdr.pdr_estimator import PDREstimator
from palkia.core.positioning.pdr.step_estimator import StepEstimator
from palkia.core.positioning.pdr.three_dimensional_estimator import (
    ThreeDimensionalEstimator,
)
from palkia.core.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.core.sensor_processing.data_loader import load_sensor_data_from_log
from palkia.core.sensor_processing.enhanced_sensor_data import EnhancedSensorData
from palkia.core.visualization import plot_trajectory
from palkia.core.visualization.floor_trajectory_plotter import plot_floor_trajectories


def main() -> None:
    acc_data = pd.read_csv("../data/raw/other/ThreeDimensional/Accelerometer.csv")
    gyro_data = pd.read_csv("../data/raw/other/ThreeDimensional/Gyroscope.csv")
    baro_data = pd.read_csv("../data/raw/other/ThreeDimensional/Barometer.csv")

    max_time = 200

    acc_data = acc_data[acc_data[TIMESTAMP] < max_time]
    gyro_data = gyro_data[gyro_data[TIMESTAMP] < max_time]

    gt_data = pd.DataFrame(
        {
            TIMESTAMP: [0, max_time],
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
                "x": gt_data[POS_X][0],
                "y": gt_data[POS_Y][0],
            },
        ),
    )

    floor_maps = {
        0: FloorMap(
            floor_name="floor_5",
            floor_map_path=FLOOR_MAP_PATH.format("floor_5", "png"),
            dx=0.01,
            dy=0.01,
        ),
        1: FloorMap(
            floor_name="floor_5",
            floor_map_path=FLOOR_MAP_PATH.format("floor_5", "png"),
            dx=0.01,
            dy=0.01,
        ),
    }

    three_demensional_estimator = ThreeDimensionalEstimator(
        pdr_estimator,
    )

    floor_segments = three_demensional_estimator.estimate_3d_trajectory_with_floors(
        baro_data,
        floor_maps,
    )

    plot_floor_trajectories(floor_segments, floor_maps)


if __name__ == "__main__":
    main()
