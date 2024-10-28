from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from palkia.positioning.floor_identification.floor_identifier import FloorInfo
    from palkia.utils.floor_map import FloorMap

from palkia.const import (
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
    *,
    ground_truth: pd.DataFrame | None = None,
    floor_map: FloorMap | None = None,
    figsize: tuple[int, int] = FIGURE_SIZE,
    scatter_size: int = SCATTER_SIZE,
    start_point_color: str = START_POINT_COLOR,
    end_point_color: str = END_POINT_COLOR,
    start_end_point_size: int = STRAT_END_POINT_SIZE,
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
        _plot_ground_truth(ground_truth)

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
) -> None:
    """Plot the ground truth trajectory."""
    plt.plot(
        ground_truth["x"],
        ground_truth["y"],
        label="Ground Truth",
    )


def _plot_floor_map(floor_map: FloorMap, target_ax: Axes | None = None) -> None:
    """Plot the floor map as a background.

    Args:
    ----
        floor_map: Floor map object to plot.
        target_ax: Matplotlib axes object to plot on. If None, uses current axes.

    """
    if target_ax is None:
        target_ax = plt.gca()

    target_ax.set_title(floor_map.floor_name)
    target_ax.imshow(
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
    acc_data: pd.DataFrame | None = None,
    gyro_data: pd.DataFrame | None = None,
    baro_data: pd.DataFrame | None = None,
) -> None:
    """Plot raw sensor data with increased spacing between subplots.

    Args:
    ----
        acc_data (pd.DataFrame): Accelerometer data.
        gyro_data (pd.DataFrame): Gyroscope data.
        baro_data (pd.DataFrame): Barometer data.

    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))  # 高さを増やす

    if acc_data is not None:
        # Plot accelerometer data
        ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_X], label="X")
        ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Y], label="Y")
        ax1.plot(acc_data[TIMESTAMP], acc_data[ACC_Z], label="Z")
        ax1.set_title("Accelerometer Data", fontsize=16)
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("Acceleration (m/s²)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.tick_params(labelsize=10)

    if gyro_data is not None:
        # Plot gyroscope data
        ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_X], label="X")
        ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_Y], label="Y")
        ax2.plot(gyro_data[TIMESTAMP], gyro_data[GYRO_Z], label="Z")
        ax2.set_title("Gyroscope Data", fontsize=16)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Angular Velocity (rad/s)", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.tick_params(labelsize=10)

    if baro_data is not None:
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


def plot_pressure_analysis(
    baro_data: pd.DataFrame,
    floor_info: dict[int, FloorInfo],
    transitions: list[dict],
) -> None:
    """Plot pressure data analysis.

    Args:
    ----
        baro_data: Barometer sensor data.
        floor_info: Dictionary mapping floor numbers to FloorInfo objects.
        transitions: List of transition events.

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # 気圧データのプロット
    ax1.plot(baro_data["ts"], baro_data["pressure"], "b-", alpha=0.5, label="Raw")
    ax1.plot(
        baro_data["ts"],
        baro_data["pressure_smoothed"],
        "r-",
        label="Smoothed",
    )

    # 階層区分の表示
    for floor, info in floor_info.items():
        p_min, p_max = info.pressure_range
        ax1.axhspan(p_min, p_max, alpha=0.2, label=f"Floor {floor}")

    # 移動区間の表示
    for t in transitions:
        ax1.axvspan(
            t["start_time"],
            t["end_time"],
            color="y",
            alpha=0.3,
        )

    ax1.set_ylabel("Pressure (hPa)")
    ax1.legend()

    # 階層推定結果の表示
    floor_numbers = []
    times = []
    for floor, info in floor_info.items():
        for start, end in info.time_intervals:
            floor_numbers.extend([floor, floor])
            times.extend([start, end])

    ax2.step(times, floor_numbers, where="post")
    ax2.set_ylabel("Floor Number")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    plt.tight_layout()


def plot_floor_trajectories(
    floor_info_dict: dict[int, FloorInfo],
    floor_maps: dict[int, FloorMap],
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """Plot trajectories for each floor with improved layout."""
    n_floors = len(floor_info_dict)
    if n_floors == 0:
        print("No floors detected!")
        return

    # figureとaxesの作成
    fig, axes = plt.subplots(
        (n_floors + 1) // 2,
        2,
        figsize=figsize,
        squeeze=False,  # 常に2D配列として返す
        constrained_layout=True,  # tight_layoutの代わりに使用
    )
    axes = axes.flatten()

    # カラーマップの準備
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, n_floors))
    last_scatter = None

    # 全てのaxesを一旦非表示に
    for ax in axes:
        ax.set_visible(False)

    for i, (floor, info) in enumerate(sorted(floor_info_dict.items())):
        ax = axes[i]
        ax.set_visible(True)

        # フロアマップの表示（存在する場合）
        if floor in floor_maps:
            floor_map = floor_maps[floor]
            ax.imshow(
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

        # 軌跡の表示
        trajectory = info.trajectory
        if not trajectory.empty:
            # 座標カラムの確認
            x_col = COORDINATE_X if COORDINATE_X in trajectory.columns else "x"
            y_col = COORDINATE_Y if COORDINATE_Y in trajectory.columns else "y"

            last_scatter = ax.scatter(
                trajectory[x_col],
                trajectory[y_col],
                c=trajectory[TIMESTAMP],
                cmap="viridis",
                s=5,
                alpha=0.6,
            )

            # 開始点と終了点をマーク
            ax.scatter(
                trajectory[x_col].iloc[0],
                trajectory[y_col].iloc[0],
                color="green",
                s=100,
                marker="^",
                label="Start",
            )
            ax.scatter(
                trajectory[x_col].iloc[-1],
                trajectory[y_col].iloc[-1],
                color="red",
                s=100,
                marker="v",
                label="End",
            )

            # 各時間区間の軌跡を表示
            for t_start, t_end in info.time_intervals:
                mask = (trajectory[TIMESTAMP] >= t_start) & (
                    trajectory[TIMESTAMP] <= t_end
                )
                segment = trajectory[mask]
                if len(segment) > 1:
                    ax.plot(
                        segment[x_col],
                        segment[y_col],
                        color=colors[i],
                        alpha=0.8,
                        linewidth=1,
                    )

            # 軸の設定
            ax.set_title(f"Floor {floor}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

    # カラーバーの追加
    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes,
            label="Time (s)",
            orientation="vertical",
            pad=0.02,
        )

    plt.show()
