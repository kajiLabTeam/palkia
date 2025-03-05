import pandas as pd

from palkia.config import (
    FLOOR_MAP_PATH,
    POS_X,
    POS_Y,
    STEP_LENGTH_MODEL_PATH,
    TIMESTAMP,
)
from palkia.core.map.floor_map import FloorMap
from palkia.core.positioning.correction.trajectory_corrector import TrajectoryCorrector
from palkia.core.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.core.positioning.pdr.pdr_estimator import PDREstimator
from palkia.core.positioning.pdr.step_estimator import StepEstimator
from palkia.core.positioning.pdr.three_dimensional_estimator import (
    ThreeDimensionalEstimator,
)
from palkia.core.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.core.sensor_processing.enhanced_sensor_data import EnhancedSensorData
from palkia.core.visualization.floor_trajectory_plotter import plot_floor_trajectories


def main() -> None:
    acc_data = pd.read_csv("../data/raw/other/ThreeDimensional/Accelerometer.csv")
    gyro_data = pd.read_csv("../data/raw/other/ThreeDimensional/Gyroscope.csv")
    baro_data = pd.read_csv("../data/raw/other/ThreeDimensional/Barometer.csv")

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

    trajectory_corrector = (
        TrajectoryCorrector.builder(pdr_estimator)
        .with_floor_map(floor_maps[0])
        .with_ground_truth(gt_data)
        .build()
    )

    three_demensional_estimator = ThreeDimensionalEstimator(
        trajectory_corrector,
    )

    floor_segments = three_demensional_estimator.estimate_3d_trajectory_with_floors(
        baro_data,
        floor_maps,
    )

    plot_floor_trajectories(floor_segments, floor_maps)


if __name__ == "__main__":
    main()
