from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Literal

import pandera as pa

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import pandas as pd
from common import utils
from PIL import Image
from scipy.signal import find_peaks

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

GIS_BASE_PATH = "../../dataset/gis/"
BEACON_LIST_PATH = GIS_BASE_PATH + "beacon_list.csv"
FLOOR_NAMES = ["FLU01", "FLU02", "FLD01"]
FOLDER_ID = "1qZBLQ66_pwRwLOy3Zj5q_qAwY_Z05HXb"


class TrajectoryEstimator:
    Axis2D = Literal["x", "y"]

    def __init__(self, log_file_directory: str, log_file_name: str):
        self.log_file_directory = log_file_directory
        self.log_file_name = log_file_name
        self.log_file_path = self.log_file_directory + self.log_file_name
        self.data = self.read_log_data()

    def read_log_data(self) -> dict:
        """Read log data from a file and return a dictionary."""
        data = defaultdict(list)
        with open(self.log_file_path) as f:
            for line in f:
                line_contents = line.rstrip("\n").split(";")
                data_type = line_contents[0]
                if data_type == "BLUE":
                    data["BLUE"].append(
                        {
                            "ts": float(line_contents[1]),
                            "bdaddress": line_contents[2],
                            "rssi": int(line_contents[4]),
                        },
                    )
                elif data_type in ["ACCE", "GYRO", "MAGN"]:
                    record = {
                        "ts": float(line_contents[1]),
                        "x": float(line_contents[3]),
                        "y": float(line_contents[4]),
                        "z": float(line_contents[5]),
                    }
                    data[data_type].append(record)
                elif data_type == "POS3":
                    data["POS3"].append(
                        {
                            "ts": float(line_contents[1]),
                            "x": float(line_contents[3]),
                            "y": float(line_contents[4]),
                            "z": float(line_contents[5]),
                            "q0": float(line_contents[6]),
                            "q1": float(line_contents[7]),
                            "q2": float(line_contents[8]),
                            "q3": float(line_contents[9]),
                            "floor_name": line_contents[10],
                        },
                    )

        return data

    def convert_to_dataframes(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert the given data dictionary into pandas DataFrames."""
        acc_df = pd.DataFrame(self.data["ACCE"])
        gyro_df = pd.DataFrame(self.data["GYRO"])
        magn_df = pd.DataFrame(self.data["MAGN"])
        pos3_df = pd.DataFrame(self.data["POS3"])
        blue_df = pd.DataFrame(self.data["BLUE"])

        # Reset index for each dataframe
        acc_df = acc_df.reset_index(drop=True)
        gyro_df = gyro_df.reset_index(drop=True)
        magn_df = magn_df.reset_index(drop=True)
        pos3_df = pos3_df.reset_index(drop=True)

        # Define schema for validation
        sensor_schema = pa.DataFrameSchema(
            {
                "ts": pa.Column(pa.Float, nullable=False),
                "x": pa.Column(pa.Float, nullable=False),
                "y": pa.Column(pa.Float, nullable=False),
                "z": pa.Column(pa.Float, nullable=False),
            },
        )

        # Validate the dataframes
        sensor_schema(acc_df)
        sensor_schema(gyro_df)
        sensor_schema(magn_df)

        return acc_df, gyro_df, magn_df, pos3_df, blue_df

    @staticmethod
    def load_floor_maps(
        floor_names: list,
        base_path: str,
        optional_file_path: str = "",
    ) -> dict[str, np.ndarray]:
        """Load floor maps from the specified base path."""
        map_dict: dict[str, np.ndarray] = {}
        for floor_name in floor_names:
            map_image_path = (
                f"{base_path}{floor_name}_0.01_0.01{optional_file_path}.bmp"
            )
            map_image = Image.open(map_image_path)
            map_dict[floor_name] = np.array(map_image, dtype=bool)
        return map_dict

    @staticmethod
    def _process_sensor_data(acc_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Process accelerometer data to detect peaks."""
        acc_df["norm"] = np.sqrt(acc_df["x"] ** 2 + acc_df["y"] ** 2 + acc_df["z"] ** 2)
        acc_df["rolling_norm"] = acc_df["norm"].rolling(10).mean()
        peaks, _ = find_peaks(acc_df["rolling_norm"], height=12, distance=10)
        return acc_df, peaks

    @staticmethod
    def estimate_trajectory(
        acc_df: pd.DataFrame,
        gyro_df: pd.DataFrame,
        *,
        ground_truth_first_point: Optional[dict[Axis2D, float]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Estimate the trajectory using accelerometer and gyroscope data."""
        if ground_truth_first_point is None:
            ground_truth_first_point = {"x": 0.0, "y": 0.0}

        acc_df, peaks = TrajectoryEstimator._process_sensor_data(acc_df)
        peek_angle = TrajectoryEstimator.convert_to_peek_angle(gyro_df, acc_df, peaks)
        return TrajectoryEstimator.convert_to_angle_from_gyro(
            gyro_df,
        ), TrajectoryEstimator.calculate_cumulative_displacement(
            peek_angle["ts"],
            peek_angle["x"],
            0.5,
            ground_truth_first_point,
        )

    @staticmethod
    def convert_to_peek_angle(
        gyro_df: pd.DataFrame,
        acc_df: pd.DataFrame,
        peaks: np.ndarray,
    ) -> pd.DataFrame:
        """Convert gyroscope data to angle at peak points."""
        angle_df = pd.DataFrame()
        angle_df["ts"] = gyro_df["ts"]
        angle_df["x"] = gyro_df["x"].cumsum() * 0.01

        return TrajectoryEstimator.match_data(angle_df, acc_df["ts"][peaks])

    @staticmethod
    def convert_to_angle_from_gyro(gyro_df: pd.DataFrame) -> pd.DataFrame:
        """Convert gyroscope data to cumulative angle."""
        angle_df = pd.DataFrame()
        angle_df["ts"] = gyro_df["ts"]
        angle_df["x"] = gyro_df["x"].cumsum() * 0.01
        angle_df["y"] = gyro_df["y"].cumsum() * 0.01
        angle_df["z"] = gyro_df["z"].cumsum() * 0.01

        return angle_df

    @staticmethod
    def match_data(angle_df: pd.DataFrame, peak_ts: pd.Series) -> pd.DataFrame:
        """Match angle data with peak timestamps."""
        matched_df = pd.DataFrame()
        for t in peak_ts:
            matched_row = angle_df[np.isclose(angle_df["ts"], t, atol=0.005)]
            matched_df = pd.concat([matched_df, matched_row])
        return matched_df

    @staticmethod
    def calculate_cumulative_displacement(
        ts: pd.Series,
        angle_data_x: pd.Series,
        step_length: float,
        initial_point: dict[str, float],
        initial_timestamp: float = 0.0,
    ) -> pd.DataFrame:
        """Calculate cumulative displacement."""
        x_displacement = step_length * np.cos(angle_data_x)
        y_displacement = step_length * np.sin(angle_data_x)

        init_data_frame = pd.DataFrame(
            {
                "ts": [initial_timestamp],
                "x_displacement": initial_point["x"],
                "y_displacement": initial_point["y"],
            },
        )

        return pd.concat(
            [
                init_data_frame,
                pd.DataFrame(
                    {
                        "ts": ts,
                        "x_displacement": x_displacement.cumsum() + initial_point["x"],
                        "y_displacement": y_displacement.cumsum() + initial_point["y"],
                    },
                ),
            ],
        )

    def run(self):
        acc_df, gyro_df, _, ground_truth_df, _ = self.convert_to_dataframes()

        true_point: dict[TrajectoryEstimator.Axis2D, float] = {
            "x": ground_truth_df["x"][0],
            "y": ground_truth_df["y"][0],
        }

        _, trajectory = self.estimate_trajectory(
            acc_df,
            gyro_df,
            ground_truth_first_point=true_point,
        )

        map_dict = self.load_floor_maps(FLOOR_NAMES, GIS_BASE_PATH)
        utils.plot_displacement_map(map_dict, "FLU01", 0.01, 0.01, trajectory)


if __name__ == "__main__":
    estimator = TrajectoryEstimator(
        log_file_directory="../../dataset/sample-trials/",
        log_file_name="4_1_51_pdr.txt",
    )
    estimator.run()
