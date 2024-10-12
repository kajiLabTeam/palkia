from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.palkia.utils.floor_map import FloorMap

from src.palkia.const import (
    ACC_X,
    ACC_Y,
    ACC_Z,
    COORDINATE_X,
    COORDINATE_Y,
    GYRO_X,
    GYRO_Y,
    GYRO_Z,
    PRESSURE,
    TIMESTAMP,
)

# Constants for plot configuration
FIGURE_SIZE = (10, 10)
SCATTER_SIZE = 5
START_POINT_COLOR = "#674598"
END_POINT_COLOR = "red"
STRAT_END_POINT_SIZE = 50
GROUND_TRUTH_COLOR = "blue"
GROUND_TRUTH_STYLE = "--"


def plot_trajectory(
    trajectory: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
    floor_map: FloorMap | None = None,
    *,
    figsize: tuple[int, int] = FIGURE_SIZE,
    scatter_size: int = SCATTER_SIZE,
    start_point_color: str = START_POINT_COLOR,
    end_point_color: str = END_POINT_COLOR,
    start_end_point_size: int = STRAT_END_POINT_SIZE,
    ground_truth_color: str = GROUND_TRUTH_COLOR,
    ground_truth_style: str = GROUND_TRUTH_STYLE,
) -> None:
    """Plot the estimated trajectory, optionally with ground truth and floor map.

    Args:
    ----
        trajectory (pd.DataFrame): Estimated trajectory data.
        ground_truth (pd.DataFrame, optional): Ground truth trajectory data.
        floor_map (FloorMap, optional): Floor map data.

        figsize (tuple[int, int], optional): Size of the figure.
        scatter_size (int, optional): Size of the scatter points.
        start_point_color (str, optional): Color of the start point.
        end_point_color (str, optional): Color of the end point.
        start_end_point_size (int, optional): Size of the start and end points.
        ground_truth_color (str, optional): Color of the ground truth line.
        ground_truth_style (str, optional): Style of the ground truth line.

    """
    plt.figure(figsize=figsize)

    _plot_estimated_trajectory(trajectory, scatter_size)
    _plot_start_end_points(
        trajectory,
        start_point_color,
        end_point_color,
        start_end_point_size,
    )

    if ground_truth is not None:
        _plot_ground_truth(ground_truth, ground_truth_color, ground_truth_style)

    if floor_map is not None:
        _plot_floor_map(floor_map)

    _set_plot_properties()
    plt.show()


def _plot_estimated_trajectory(
    trajectory: pd.DataFrame,
    scatter_size: int = SCATTER_SIZE,
) -> None:
    """Plot the estimated trajectory with a color gradient based on time."""
    scatter = plt.scatter(
        trajectory[COORDINATE_X],
        trajectory[COORDINATE_Y],
        c=trajectory[TIMESTAMP],
        cmap="rainbow",
        s=scatter_size,
    )
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("time(s)", fontsize=12)


def _plot_start_end_points(
    trajectory: pd.DataFrame,
    start_point_color: str = START_POINT_COLOR,
    end_point_color: str = END_POINT_COLOR,
    start_end_point_size: int = STRAT_END_POINT_SIZE,
) -> None:
    """Plot the start and end points of the trajectory."""
    plt.scatter(
        trajectory[COORDINATE_X].iloc[0],
        trajectory[COORDINATE_Y].iloc[0],
        c=start_point_color,
        s=start_end_point_size,
        label="Start",
    )
    plt.scatter(
        trajectory[COORDINATE_X].iloc[-1],
        trajectory[COORDINATE_Y].iloc[-1],
        c=end_point_color,
        s=start_end_point_size,
        label="End",
    )


def _plot_ground_truth(
    ground_truth: pd.DataFrame,
    color: str = GROUND_TRUTH_COLOR,
    style: str = GROUND_TRUTH_STYLE,
) -> None:
    """Plot the ground truth trajectory."""
    plt.plot(
        ground_truth["x"],
        ground_truth["y"],
        f"{color}{style}",
        label="Ground Truth",
    )


def _plot_floor_map(floor_map: FloorMap) -> None:
    """Plot the floor map as a background."""
    plt.title(floor_map.floor_name)
    plt.imshow(
        np.rot90(floor_map.floor_map_data),
        extent=(
            0,
            floor_map.floor_map_data.shape[0] * floor_map.dx,
            0,
            floor_map.floor_map_data.shape[1] * floor_map.dy,
        ),
        cmap="binary",
        alpha=0.5,
    )


def _set_plot_properties() -> None:
    """Set general properties of the plot."""
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory")
    plt.legend()
    plt.grid()
    plt.axis("equal")


def plot_sensor_data(
    acc_data: pd.DataFrame,
    gyro_data: pd.DataFrame,
    baro_data: pd.DataFrame,
) -> None:
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
    ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Z], label="Z")
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
