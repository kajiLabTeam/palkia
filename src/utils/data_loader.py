from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from src.const import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    BLE_ADDRESS,
    FLOOR_NAME,
    GYRO_X,
    GYRO_Y,
    GYRO_Z,
    POS_X,
    POS_Y,
    POS_Z,
    PRESSURE,
    QUATERNION_0,
    QUATERNION_1,
    QUATERNION_2,
    QUATERNION_3,
    RSSI,
    TIMESTAMP,
)


def read_log_data(log_file_path: str) -> dict:
    data = defaultdict(list)
    with open(log_file_path) as f:
        for line in f:
            line_contents = line.rstrip("\n").split(";")
            data_type = line_contents[0]
            if data_type == "BLUE":
                data["BLUE"].append(
                    {
                        TIMESTAMP: float(line_contents[1]),
                        BLE_ADDRESS: line_contents[2],
                        RSSI: int(line_contents[4]),
                    },
                )
            elif data_type in ["ACCE", "GYRO", "MAGN", "BARO"]:
                record = {
                    TIMESTAMP: float(line_contents[1]),
                    ACC_X if data_type == "ACCE" else GYRO_X: float(line_contents[3]),
                    ACC_Y if data_type == "ACCE" else GYRO_Y: float(line_contents[4]),
                    ACC_Z if data_type == "ACCE" else GYRO_Z: float(line_contents[5]),
                }
                if data_type == "BARO":
                    record[PRESSURE] = float(line_contents[3])
                data[data_type].append(record)
            elif data_type == "POS3":
                data["POS3"].append(
                    {
                        TIMESTAMP: float(line_contents[1]),
                        POS_X: float(line_contents[3]),
                        POS_Y: float(line_contents[4]),
                        POS_Z: float(line_contents[5]),
                        QUATERNION_0: float(line_contents[6]),
                        QUATERNION_1: float(line_contents[7]),
                        QUATERNION_2: float(line_contents[8]),
                        QUATERNION_3: float(line_contents[9]),
                        FLOOR_NAME: line_contents[10],
                    },
                )

    # Convert lists of dictionaries to DataFrames
    for key in data:
        data[key] = pd.DataFrame(data[key])

    return data


def load_sensor_data_from_log(log_file_path: str):
    """Load sensor data from a log file.

    Args:
    ----
        log_file_path (str): Path to the log file.

    Returns:
    -------
        tuple: Tuple containing DataFrames for accelerometer, gyroscope, magnetometer,
               barometer, ground truth, and BLE scan data.

    """
    data = read_log_data(log_file_path)

    acc_df = data.get("ACCE", pd.DataFrame())
    gyro_df = data.get("GYRO", pd.DataFrame())
    mag_df = data.get("MAGN", pd.DataFrame())
    baro_df = data.get("BARO", pd.DataFrame())  # 気圧データを追加
    gt_df = data.get("POS3", pd.DataFrame())
    ble_df = data.get("BLUE", pd.DataFrame())

    return acc_df, gyro_df, mag_df, baro_df, gt_df, ble_df


def load_floor_map(
    floor_map_path: str,
) -> np.ndarray:
    """Load a floor map from the specified base path.

    Args:
    ----
        floor_name (str): Name of the floor.
        base_path (str): Base path of the floor maps.
        optional_file_path (str): Optional file path to append to the base path.

    Returns:
    -------
        np.ndarray: Floor map.

    """
    map_image_path = floor_map_path
    map_image = Image.open(map_image_path)
    return np.array(map_image, dtype=bool)


def load_floor_maps(
    floor_names: list,
    base_path: str,
    optional_file_path: str = "",
) -> dict[str, np.ndarray]:
    """Load floor maps from the specified base path.

    Args:
    ----
        floor_names (list): List of floor names.
        base_path (str): Base path of the floor maps.
        optional_file_path (str): Optional file path to append to the base path.

    Returns:
    -------
        dict[str, np.ndarray]: Dictionary mapping floor names to floor maps.

    """
    map_dict: dict[str, np.ndarray] = {}
    for floor_name in floor_names:
        map_image_path = f"{base_path}{floor_name}_0.01_0.01{optional_file_path}.bmp"
        map_image = Image.open(map_image_path)
        map_dict[floor_name] = np.array(map_image, dtype=bool)
        print(map_dict[floor_name].shape)
    return map_dict
