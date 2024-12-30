from palkia.config import (
    FLOOR_MAP_PATH,
    FLOOR_NAME,
    LOG_FILE_PATH,
    POS_X,
    POS_Y,
)
from palkia.core.map.floor_map import FloorMap
from palkia.core.positioning.correction.trajectory_corrector import TrajectoryCorrector
from palkia.core.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.core.positioning.pdr.pdr_estimator import PDREstimator
from palkia.core.positioning.pdr.step_estimator import StepEstimator
from palkia.core.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.core.sensor_processing.data_loader import load_sensor_data_from_log
from palkia.core.sensor_processing.enhanced_sensor_data import EnhancedSensorData
from palkia.core.visualization import plot_trajectory


def main() -> None:
    acc_data, gyro_data, baro_data, _, gt_data, ble_data = load_sensor_data_from_log(
        LOG_FILE_PATH,
    )

    # PDR推定器の初期化
    pdr_estimator = PDREstimator(
        EnhancedSensorData(
            acc_data,
            gyro_data,
        ),
        StepEstimator(),
        OrientationEstimator(),
        TrajectoryCalculator(
            # flip_vertical=True,
            initial_point={
                "x": gt_data.loc[0, POS_X],
                "y": gt_data.loc[0, POS_Y],
            },
        ),
    )

    # 軌跡の推定
    floor_name = gt_data.loc[0, FLOOR_NAME]
    floor_map = FloorMap(
        floor_name=floor_name,
        floor_map_path=FLOOR_MAP_PATH.format(floor_name, "bmp"),
        dx=0.01,
        dy=0.01,
    )

    trajectory_corrector = (
        TrajectoryCorrector.builder(pdr_estimator)
        .with_floor_map(floor_map)
        .with_ble_data(ble_data)
        .with_ground_truth(gt_data)
        .build()
    )

    correct_trajectory = trajectory_corrector.estimate_and_correct_trajectory()

    plot_trajectory(correct_trajectory, floor_map=floor_map)


if __name__ == "__main__":
    main()
