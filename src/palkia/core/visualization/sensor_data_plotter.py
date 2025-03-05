# sensor_data_plotter.py
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from palkia.config import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    GYRO_X,
    GYRO_Y,
    GYRO_Z,
    PRESSURE,
    TIMESTAMP,
)

from .plot_utils import setup_axis

if TYPE_CHECKING:
    import pandas as pd


def plot_sensor_data(
    acc_data: pd.DataFrame | None = None,
    gyro_data: pd.DataFrame | None = None,
    baro_data: pd.DataFrame | None = None,
) -> None:
    """センサーデータのプロット.

    Args:
    ----
        acc_data: 加速度センサーデータ
        gyro_data: ジャイロセンサーデータ
        baro_data: 気圧センサーデータ

    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

    if acc_data is not None:
        ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_X], label="X")
        ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Y], label="Y")
        ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Z], label="Z")
        setup_axis(ax1, "Accelerometer Data", ylabel="Acceleration (m/s²)", aspect=None)
        ax1.legend(fontsize=10)
        ax1.tick_params(labelsize=10)

    if gyro_data is not None:
        ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_X], label="X")
        ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_Y], label="Y")
        ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_Z], label="Z")
        setup_axis(
            ax2, "Gyroscope Data", ylabel="Angular Velocity (rad/s)", aspect=None
        )
        ax2.legend(fontsize=10)
        ax2.tick_params(labelsize=10)

    if baro_data is not None and PRESSURE in baro_data.columns:
        ax3.plot(baro_data[TIMESTAMP], baro_data[PRESSURE])
        setup_axis(ax3, "Barometer Data", ylabel="Pressure (hPa)", aspect=None)
        ax3.tick_params(labelsize=10)
    else:
        ax3.text(
            0.5,
            0.5,
            "No barometer data available",
            ha="center",
            va="center",
            fontsize=14,
        )

    fig.subplots_adjust(hspace=0.4)
    plt.show()
