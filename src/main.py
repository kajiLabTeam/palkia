from palkia.const import (
    DEFAULT_STEP_LENGTH,
    FLOOR_MAP_PATH,
    FLOOR_NAME,
    LOG_FILE_PATH,
    POS_X,
    POS_Y,
    STEP_LENGTH_MODEL_PATH,
)
from palkia.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.positioning.pdr.pdr_estimator import PDREstimator
from palkia.positioning.pdr.step_estimator import StepEstimator
from palkia.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.utils.data_loader import load_sensor_data_from_log
from palkia.utils.floor_map import FloorMap
from palkia.utils.visualizer import plot_sensor_data, plot_trajectory


def main() -> None:
    acc_data, gyro_data, baro_data, mag_data, gt_data, ble_data = (
        load_sensor_data_from_log(
            LOG_FILE_PATH,
        )
    )

    # センサーデータの可視化
    plot_sensor_data(acc_data, gyro_data)

    #  PDRコンポーネントの初期化
    step_estimator = StepEstimator(
        step_length=DEFAULT_STEP_LENGTH,
        step_length_model_path=STEP_LENGTH_MODEL_PATH,
    )
    orientation_estimator = OrientationEstimator()
    trajectory_calculator = TrajectoryCalculator(
        initial_point={
            "x": gt_data[POS_X][0],
            "y": gt_data[POS_Y][0],
        },
    )

    # PDR推定器の初期化
    pdr_estimator = PDREstimator(
        step_estimator,
        orientation_estimator,
        trajectory_calculator,
    )

    # 軌跡の推定
    trajectory = pdr_estimator.estimate_trajectory(acc_data, gyro_data)

    floor_map = FloorMap(
        floor_name=gt_data[FLOOR_NAME][0],
        floor_map_path=FLOOR_MAP_PATH.format(gt_data[FLOOR_NAME][0]),
        dx=0.01,
        dy=0.01,
    )
    #  推定軌跡の可視化
    plot_trajectory(trajectory, floor_map=floor_map)

    # # Ground truthとの比較（オプション）
    if not gt_data.empty:
        plot_trajectory(trajectory, ground_truth=gt_data)


if __name__ == "__main__":
    main()
