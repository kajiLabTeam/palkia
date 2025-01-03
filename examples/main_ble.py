import pandas as pd

from palkia.config import (
    FLOOR_MAP_PATH,
    FLOOR_NAME,
    LOG_FILE_PATH,
    POS_X,
    POS_Y,
)
from palkia.config.path import BEACON_FP_PATH, BEACON_LIST_PATH
from palkia.core.map.floor_map import FloorMap
from palkia.core.positioning.correction import (
    DriftCorrector,
    MapMatchCorrector,
    WirelessSignalCorrector,
)
from palkia.core.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.core.positioning.pdr.pdr_estimator import PDREstimator
from palkia.core.positioning.pdr.step_estimator import StepEstimator
from palkia.core.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.core.sensor_processing.data_loader import load_sensor_data_from_log
from palkia.core.sensor_processing.enhanced_sensor_data import EnhancedSensorData
from palkia.core.visualization import plot_trajectory


def main() -> None:
    acc_data, gyro_data, baro_data, mag_data, gt_data, signal_realtime_scans = (
        load_sensor_data_from_log(
            LOG_FILE_PATH,
        )
    )

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
                "x": gt_data[POS_X].iloc[0],
                "y": gt_data[POS_Y].iloc[0],
            },
        ),
    )

    # 軌跡の推定
    floor_name = gt_data[FLOOR_NAME].iloc[0]
    floor_map = FloorMap(
        floor_name=floor_name,
        floor_map_path=FLOOR_MAP_PATH.format(floor_name, "bmp"),
        dx=0.01,
        dy=0.01,
    )

    trajectory = pdr_estimator.estimate_trajectory()

    correct_drift_trajectory = DriftCorrector(
        config={}, pdr_estimator=pdr_estimator, gt_data=gt_data
    ).correct_drift()

    ble_fp = pd.read_csv(BEACON_FP_PATH)

    ble_correction_trajectory = WirelessSignalCorrector(
        signal_realtime_scans=signal_realtime_scans,
        rssi_threshold=-75,
        transmitter_positions=pd.read_csv(BEACON_LIST_PATH),
    ).correct_initial_direction_with_transmitter_positions(
        correct_drift_trajectory,
    )

    plot_trajectory(ble_correction_trajectory, floor_map=floor_map)

    walkable_trajectory = MapMatchCorrector(
        config={}, pdr_estimator=pdr_estimator, floor_map=floor_map
    ).correct_unwalkable_points(ble_correction_trajectory)

    plot_trajectory(walkable_trajectory, floor_map=floor_map)


if __name__ == "__main__":
    main()
