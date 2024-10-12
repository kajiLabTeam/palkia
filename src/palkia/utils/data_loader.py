from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.palkia.const import (
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


def read_log_data(log_file_path: str) -> dict[str, pd.DataFrame]:
    data: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)

    with Path(log_file_path).open() as fp:
        for line in fp:
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
                data[data_type].append(record)  # type: ignore[attr-defined]
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
    return {key: pd.DataFrame(value) for key, value in data.items()}


def load_sensor_data_from_log(
    log_file_path: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
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
