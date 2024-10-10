import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from src.const import FLOOR_MAP_PATH, FLOOR_NAME, LOG_FILE_PATH, POS_X, POS_Y
from src.pdr.orientation_estimator import OrientationEstimator
from src.pdr.pdr_estimator import PDREstimator
from src.pdr.step_detector import StepDetector
from src.pdr.trajectory_calculator import TrajectoryCalculator
from src.utils.data_loader import load_sensor_data_from_log
from src.utils.floor_map import FloorMap
from src.utils.visualizer import plot_sensor_data, plot_trajectory


def main() -> None:
    acc_data, gyro_data, baro_data, mag_data, gt_data, ble_data = (
        load_sensor_data_from_log(
            LOG_FILE_PATH,
        )
    )
    # センサーデータの可視化
    # plot_sensor_data(acc_data, gyro_data, mag_data)

    #  PDRコンポーネントの初期化
    step_detector = StepDetector()
    orientation_estimator = OrientationEstimator()
    trajectory_calculator = TrajectoryCalculator(
        step_length=0.5,
        initial_point={
            "x": gt_data[POS_X][0],
            "y": gt_data[POS_Y][0],
        },
    )

    # PDR推定器の初期化
    pdr_estimator = PDREstimator(
        step_detector,
        orientation_estimator,
        trajectory_calculator,
    )
    # 軌跡の推定
    trajectory = pdr_estimator.estimate_trajectory(acc_data, gyro_data)

    floor_map = FloorMap(
        floor_map_path=FLOOR_MAP_PATH.format(gt_data[FLOOR_NAME][0]),
        dx=0.01,
        dy=0.01,
    )
    #  推定軌跡の可視化
    plot_trajectory(trajectory, floor_map=floor_map)

    # # Ground truthとの比較（オプション）
    # if not gt_data.empty:
    #     plot_trajectory(trajectory, ground_truth=gt_data)


if __name__ == "__main__":
    main()
