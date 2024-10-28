from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from palkia.const import COORDINATE_X, COORDINATE_Y, TIMESTAMP

from .plot_utils import (
    END_POINT_COLOR,
    SCATTER_SIZE,
    START_END_POINT_SIZE,
    START_POINT_COLOR,
)

if TYPE_CHECKING:
    import pandas as pd

    from palkia.utils.floor_map import FloorMap


def _plot_estimated_trajectory(
    trajectory: pd.DataFrame,
    scatter_size: int = SCATTER_SIZE,
) -> None:
    """軌跡のプロット."""
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
    start_end_point_size: int = START_END_POINT_SIZE,
) -> None:
    """開始点と終了点のプロット."""
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
    """真値軌跡のプロット."""
    plt.plot(
        ground_truth["x"],
        ground_truth["y"],
        label="Ground Truth",
    )


def _plot_floor_map(floor_map: FloorMap) -> None:
    """フロアマップのプロット."""
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
