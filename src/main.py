import pandas as pd

from palkia.config import (
    FLOOR_MAP_PATH,
    FLOOR_NAME,
    LOG_FILE_PATH,
    POS_X,
    POS_Y,
    STEP_LENGTH_MODEL_PATH,
    TIMESTAMP,
)
from palkia.positioning.correction import DriftCorrector, MapMatcher
from palkia.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.positioning.pdr.pdr_estimator import PDREstimator
from palkia.positioning.pdr.step_estimator import StepEstimator
from palkia.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.utils.data_loader import load_sensor_data_from_log
from palkia.utils.enhanced_sensor_data import EnhancedSensorData
from palkia.utils.floor_map import FloorMap
from palkia.utils.visualization import plot_trajectory


def main() -> None:
    acc_data, gyro_data, baro_data, mag_data, gt_data, ble_data = (
        load_sensor_data_from_log(
            LOG_FILE_PATH,
        )
    )

    # センサーデータの可視化
    # plot_sensor_data(acc_data, gyro_data)

    # PDR推定器の初期化
    pdr_estimator = PDREstimator(
        EnhancedSensorData(
            acc_data,
            gyro_data,
        ),
        StepEstimator(
            # step_length_model_path=STEP_LENGTH_MODEL_PATH,
        ),
        OrientationEstimator(),
        TrajectoryCalculator(
            # flip_vertical=True,
            initial_point={
                "x": gt_data[POS_X][0],
                "y": gt_data[POS_Y][0],
            },
        ),
    )
    # 軌跡の推定
    floor_name = gt_data[FLOOR_NAME][0]
    floor_map = FloorMap(
        floor_name=floor_name,
        floor_map_path=FLOOR_MAP_PATH.format(floor_name, "bmp"),
        dx=0.01,
        dy=0.01,
    )
    trajectory = pdr_estimator.estimate_trajectory()

    plot_trajectory(trajectory, floor_map=floor_map)

    correct_drift_trajectory = DriftCorrector(
        config={}, pdr_estimator=pdr_estimator, gt_data=gt_data
    ).correct()

    # 推定軌跡の可視化
    # plot_trajectory(correct_drift_trajectory, floor_map=floor_map)

    correct_map_matching_trajectory = MapMatcher(
        config={}, pdr_estimator=pdr_estimator, floor_map=floor_map
    ).correct_initial_direction()

    plot_trajectory(correct_map_matching_trajectory, floor_map=floor_map)

    # walkable_trajectory = MapMatcher(
    #     config={}, pdr_estimator=pdr_estimator, floor_map=floor_map
    # ).correct_unwalkable_points(correct_map_matching_trajectory)
    #
    # plot_trajectory(walkable_trajectory, floor_map=floor_map)

    # # Ground truthとの比較（オプション）
    # if not gt_data.empty:
    #     plot_trajectory(trajectory, ground_truth=gt_data)


if __name__ == "__main__":
    main()
