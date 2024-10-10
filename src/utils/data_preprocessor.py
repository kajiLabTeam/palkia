from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from src.const import (
    TIMESTAMP,
)


def preprocess_data(
    data: pd.DataFrame,
    lowpass_freq: float = 5.0,
    sampling_rate: float = 100.0,
) -> pd.DataFrame:
    """Preprocess sensor data by applying a low-pass filter and resampling.

    Args:
    ----
        data (pd.DataFrame): Input sensor data.
        lowpass_freq (float): Cutoff frequency for the low-pass filter.
        sampling_rate (float): Desired sampling rate after resampling.

    Returns:
    -------
        pd.DataFrame: Preprocessed sensor data.

    """
    # Apply low-pass filter
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = lowpass_freq / nyquist_freq
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)

    filtered_data = pd.DataFrame()
    for column in data.columns:
        if column != TIMESTAMP:
            filtered_data[column] = signal.filtfilt(b, a, data[column])

    filtered_data[TIMESTAMP] = data[TIMESTAMP]

    # Resample data
    return (
        filtered_data.set_index(TIMESTAMP)
        .resample(f"{1/sampling_rate}S")
        .mean()
        .reset_index()
    )


def align_sensor_data(
    acc_data: pd.DataFrame,
    gyro_data: pd.DataFrame,
    baro_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align different sensor data based on their timestamps.

    Args:
    ----
        acc_data (pd.DataFrame): Accelerometer data.
        gyro_data (pd.DataFrame): Gyroscope data.
        baro_data (pd.DataFrame): Barometer data.

    Returns:
    -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Aligned sensor data.

    """
    # Implement time alignment logic here
    # This could involve interpolation or resampling to a common time base
    # For simplicity, let's assume we resample all data to the accelerometer's timestamps

    common_time = acc_data[TIMESTAMP]

    aligned_gyro = (
        gyro_data.set_index(TIMESTAMP)
        .reindex(common_time, method="nearest")
        .reset_index()
    )
    aligned_baro = (
        baro_data.set_index(TIMESTAMP)
        .reindex(common_time, method="nearest")
        .reset_index()
    )

    return acc_data, aligned_gyro, aligned_baro


def match_data(
    data_df: pd.DataFrame,
    reference_times: pd.Series,
    time_column: str = "ts",
    tolerance: float = 0.005,
) -> pd.DataFrame:
    """時系列データから指定された時間に最も近いデータポイントを抽出します。.

    Args:
    ----
        data_df (pd.DataFrame): 元のデータフレーム
        reference_times (pd.Series): 抽出したい参照時間のシリーズ
        time_column (str): データフレーム内の時間列の名前
        tolerance (float): 時間の一致を判断する際の許容誤差 (秒)

    Returns:
    -------
        pd.DataFrame: 抽出されたデータポイントを含むデータフレーム

    """
    matched_df = pd.DataFrame()
    for t in reference_times:
        matched_row = data_df[np.isclose(data_df[time_column], t, atol=tolerance)]
        matched_df = pd.concat([matched_df, matched_row])
    return matched_df.reset_index(drop=True)
