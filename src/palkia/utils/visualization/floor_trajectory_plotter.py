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

plt.rcParams["font.family"] = "Hiragino Sans"  # Mac の場合
# フォントサイズの基本設定
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

# フォントサイズをグローバルに設定
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=BIGGER_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=MEDIUM_SIZE)


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
        # label="Start",
    )
    ax.scatter(
        trajectory[x_col].iloc[-1],
        trajectory[y_col].iloc[-1],
        color="red",
        s=100,
        marker="v",
        label="End",
    )

    # # 時間区間ごとの軌跡
    # for t_start, t_end in time_intervals:
    #     mask = (trajectory[TIMESTAMP] >= t_start) & (trajectory[TIMESTAMP] <= t_end)
    #     segment = trajectory[mask]
    #     if len(segment) > 1:
    #         ax.plot(segment[x_col], segment[y_col], color=color, alpha=0.8, linewidth=1)

    return scatter


def plot_floor_trajectories(
    floor_info_dict: dict[int, FloorInfo],
    floor_maps: dict[int, FloorMap],
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """階層ごとの軌跡をプロット."""
    n_floors = len(floor_info_dict)
    if n_floors == 0:
        logger.warning("No floors detected!")
        return

    # figureを作成し、カラーバー用のスペースを確保
    fig = plt.figure(figsize=figsize)

    # グリッド状のレイアウトを作成（右側にカラーバー用のスペースを確保）
    gs = fig.add_gridspec(
        (n_floors + 1) // 2,
        2,
        width_ratios=[1, 1],
        left=0.1,  # 左マージン
        right=0.85,  # 右マージン（カラーバー用にスペースを空ける）
        hspace=0.3,  # サブプロット間の垂直方向の間隔
        wspace=0.2,  # サブプロット間の水平方向の間隔
    )

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range((n_floors + 1) // 2 * 2)]
    colors = create_colormap(n_floors)
    last_scatter: PathCollection | None = None

    # 全体の時間範囲を取得
    all_timestamps = []
    for info in floor_info_dict.values():
        all_timestamps.extend(info.trajectory[TIMESTAMP].tolist())
    time_min, time_max = min(all_timestamps), max(all_timestamps)

    # 全てのaxesを一旦非表示に
    for ax in axes:
        ax.set_visible(False)

    for i, (floor, info) in enumerate(sorted(floor_info_dict.items())):
        ax = axes[i]
        ax.set_visible(True)

        if floor in floor_maps:
            _plot_floor_map(floor_maps[floor], ax)

        if not info.trajectory.empty:
            # 正規化された時間を使用
            normalized_time = (info.trajectory[TIMESTAMP] - time_min) / (
                time_max - time_min
            )

            last_scatter = ax.scatter(
                info.trajectory[COORDINATE_X],
                info.trajectory[COORDINATE_Y],
                c=normalized_time,
                cmap="rainbow",
                s=5,
                alpha=1,
                vmin=0,
                vmax=1,
            )

            # # 開始点と終了点
            # ax.scatter(
            #     info.trajectory[COORDINATE_X].iloc[0],
            #     info.trajectory[COORDINATE_Y].iloc[0],
            #     color="green",
            #     s=100,
            #     marker="^",
            #     # label="Start",
            # )

            # ax.scatter(
            #     info.trajectory[COORDINATE_X].iloc[-1],
            #     info.trajectory[COORDINATE_Y].iloc[-1],
            #     color="red",
            #     s=100,
            #     marker="v",
            #     label="End",
            # )

            setup_axis(ax, f"フロア {floor+4}階")
            ax.legend(loc="upper right")

    if last_scatter is not None:
        # カラーバーを図の右側に配置
        cax = fig.add_axes((0.87, 0.15, 0.02, 0.7))  # 位置とサイズを調整
        cbar = fig.colorbar(
            last_scatter,
            cax=cax,
            label="Time (s)",
        )
        # 元のタイムスタンプに戻した目盛りを設定
        ticks = list(np.linspace(0, 1, 5))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(
            [f"{t * (time_max - time_min) + time_min:.1f}" for t in ticks]
        )

    plt.show()
