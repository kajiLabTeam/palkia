import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.const import (
    ACC_X,
    ACC_Y,
    COORDINATE_X,
    COORDINATE_Y,
    GYRO_X,
    GYRO_Y,
    GYRO_Z,
    PRESSURE,
    TIMESTAMP,
)


def plot_trajectory(
    trajectory: pd.DataFrame,
    ground_truth: pd.DataFrame = None,
    floor_map: np.ndarray = None,
):
    """Plot the estimated trajectory, optionally with ground truth and floor map.

    Args:
    ----
        trajectory (pd.DataFrame): Estimated trajectory data.
        ground_truth (pd.DataFrame, optional): Ground truth trajectory data.
        floor_map (np.ndarray, optional): Floor map data.

    """
    plt.figure(figsize=(10, 10))

    if floor_map is not None:
        plt.imshow(
            np.rot90(floor_map),
            extent=(0, floor_map.shape[0], 0, floor_map.shape[1]),
            cmap="binary",
            alpha=0.5,
        )

    plt.plot(
        trajectory[COORDINATE_X],
        trajectory[COORDINATE_Y],
        "r-",
        label="Estimated",
    )
    plt.scatter(
        trajectory[COORDINATE_X].iloc[0],
        trajectory[COORDINATE_Y].iloc[0],
        c="g",
        s=100,
        label="Start",
    )
    plt.scatter(
        trajectory[COORDINATE_X].iloc[-1],
        trajectory[COORDINATE_Y].iloc[-1],
        c="r",
        s=100,
        label="End",
    )

    if ground_truth is not None:
        plt.plot(ground_truth["x"], ground_truth["y"], "b--", label="Ground Truth")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


def plot_sensor_data(
    acc_data: pd.DataFrame,
    gyro_data: pd.DataFrame,
    baro_data: pd.DataFrame,
):
    """Plot raw sensor data with increased spacing between subplots.

    Args:
    ----
        acc_data (pd.DataFrame): Accelerometer data.
        gyro_data (pd.DataFrame): Gyroscope data.
        baro_data (pd.DataFrame): Barometer data.

    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))  # 高さを増やす

    # Plot accelerometer data
    ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_X], label="X")
    ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Y], label="Y")
    ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Y], label="Z")
    ax1.set_title("Accelerometer Data", fontsize=16)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Acceleration (m/s²)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=10)

    # Plot gyroscope data
    ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_X], label="X")
    ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_Y], label="Y")
    ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_Z], label="Z")
    ax2.set_title("Gyroscope Data", fontsize=16)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=10)

    # Plot barometer data
    if PRESSURE in baro_data.columns:
        ax3.plot(baro_data[TIMESTAMP], baro_data[PRESSURE])
        ax3.set_title("Barometer Data", fontsize=16)
        ax3.set_xlabel("Time", fontsize=12)
        ax3.set_ylabel("Pressure (hPa)", fontsize=12)
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

    # サブプロット間の間隔を調整
    fig.subplots_adjust(hspace=0.4)  # 垂直方向の間隔を増やす

    plt.show()
