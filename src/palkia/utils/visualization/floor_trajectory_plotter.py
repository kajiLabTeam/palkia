# floor_trajectory_plotter.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from palkia.const import COORDINATE_X, COORDINATE_Y, TIMESTAMP

from .plot_utils import create_colormap, setup_axis, setup_subplots

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection

    from palkia.positioning.floor_identification import FloorInfo
    from palkia.utils.floor_map import FloorMap

logger = logging.getLogger(__name__)


def _plot_floor_map(floor_map: FloorMap, ax: Axes) -> None:
    """フロアマップのプロット."""
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


def _plot_floor_trajectory(
    ax: Axes,
    trajectory: pd.DataFrame,
    time_intervals: list[tuple[float, float]],
    color: np.ndarray,
) -> PathCollection:
    """1つのフロアの軌跡をプロット.

    Returns
    -------
        PathCollection: scatterプロットのコレクション

    """
    x_col = COORDINATE_X if COORDINATE_X in trajectory.columns else "x"
    y_col = COORDINATE_Y if COORDINATE_Y in trajectory.columns else "y"

    scatter = ax.scatter(
        trajectory[x_col],
        trajectory[y_col],
        c=trajectory[TIMESTAMP],
        cmap="rainbow",
        s=5,
        alpha=1,
    )

    # 開始点と終了点
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

    # 時間区間ごとの軌跡
    for t_start, t_end in time_intervals:
        mask = (trajectory[TIMESTAMP] >= t_start) & (trajectory[TIMESTAMP] <= t_end)
        segment = trajectory[mask]
        if len(segment) > 1:
            ax.plot(segment[x_col], segment[y_col], color=color, alpha=0.8, linewidth=1)

    return scatter


def plot_floor_trajectories(
    floor_info_dict: dict[int, FloorInfo],
    floor_maps: dict[int, FloorMap],
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """階層ごとの軌跡をプロット.

    Args:
    ----
        floor_info_dict: 階層ごとの情報を含む辞書
        floor_maps: 階層ごとのマップを含む辞書
        figsize: 図のサイズ

    """
    n_floors = len(floor_info_dict)
    if n_floors == 0:
        logger.warning("No floors detected!")
        return

    fig, axes = setup_subplots((n_floors + 1) // 2, 2, figsize)
    colors = create_colormap(n_floors)
    last_scatter: PathCollection | None = None

    # 全てのaxesを一旦非表示に
    for ax in axes:
        ax.set_visible(False)

    for i, (floor, info) in enumerate(sorted(floor_info_dict.items())):
        ax = axes[i]
        ax.set_visible(True)

        if floor in floor_maps:
            _plot_floor_map(floor_maps[floor], ax)

        if not info.trajectory.empty:
            last_scatter = _plot_floor_trajectory(
                ax, info.trajectory, info.time_intervals, colors[i]
            )
            setup_axis(ax, f"Floor {floor}")
            ax.legend(loc="upper right")

    if last_scatter is not None:
        fig.colorbar(
            last_scatter,
            ax=axes,
            label="Time (s)",
            orientation="vertical",
            pad=0.02,
        )

    plt.show()
