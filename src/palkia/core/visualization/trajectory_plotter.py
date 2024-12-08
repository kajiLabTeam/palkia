from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from palkia.config import COORDINATE_X, COORDINATE_Y, TIMESTAMP

from .plot_utils import (
    END_POINT_COLOR,
    SCATTER_SIZE,
    START_END_POINT_SIZE,
    START_POINT_COLOR,
    setup_axis,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from palkia.core.map.floor_map import FloorMap


def _add_floor_map(ax: Axes, floor_map: FloorMap) -> None:
    """フロアマップをプロット."""
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
    ax.set_title(floor_map.floor_name)


def plot_trajectory(
    trajectory: pd.DataFrame,
    *,
    ground_truth: pd.DataFrame | None = None,
    floor_map: FloorMap | None = None,
    figsize: tuple[int, int] = (12, 12),
    scatter_size: int = SCATTER_SIZE,
    start_point_color: str = START_POINT_COLOR,
    end_point_color: str = END_POINT_COLOR,
    start_end_point_size: int = START_END_POINT_SIZE,
) -> None:
    """単一の軌跡をプロット."""
    fig, ax = plt.subplots(figsize=figsize)

    if floor_map is not None:
        _add_floor_map(ax, floor_map)

    # メインの軌跡
    scatter = ax.scatter(
        trajectory[COORDINATE_X],
        trajectory[COORDINATE_Y],
        c=trajectory[TIMESTAMP],
        cmap="rainbow",
        s=scatter_size,
    )

    # 開始点と終了点
    ax.scatter(
        trajectory[COORDINATE_X].iloc[0],
        trajectory[COORDINATE_Y].iloc[0],
        color=start_point_color,
        s=start_end_point_size,
        marker="o",
        label="Start",
    )
    ax.scatter(
        trajectory[COORDINATE_X].iloc[-1],
        trajectory[COORDINATE_Y].iloc[-1],
        color=end_point_color,
        s=start_end_point_size,
        marker="o",
        label="End",
    )

    if ground_truth is not None:
        ax.plot(
            ground_truth["x"],
            ground_truth["y"],
            "b--",
            label="Ground Truth",
            alpha=0.7,
        )

    if floor_map is not None:
        setup_axis(ax, floor_map.floor_name)
    else:
        setup_axis(ax, "Trajectory")

    ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, label="Time (s)")
    plt.show()
