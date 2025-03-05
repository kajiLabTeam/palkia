from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# 共通の定数
FIGURE_SIZE = (10, 10)
SCATTER_SIZE = 5
START_POINT_COLOR = "#674598"
END_POINT_COLOR = "red"
START_END_POINT_SIZE = 50
GROUND_TRUTH_COLOR = "blue"
GROUND_TRUTH_STYLE = "--"


def setup_axis(
    ax: Axes,
    title: str,
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    aspect: float | Literal["auto", "equal"] | None = "equal",
) -> None:
    """軸の基本設定.

    Args:
    ----
        ax: 設定対象の軸
        title: グラフのタイトル
        xlabel: x軸のラベル
        ylabel: y軸のラベル
        aspect: アスペクト比。"auto", "equal"、数値、またはNone

    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if aspect is not None:
        ax.set_aspect(aspect)
    ax.grid(visible=True, alpha=0.3)


def create_colormap(n_colors: int) -> np.ndarray:
    """カラーマップの生成.

    Args:
    ----
        n_colors: 生成する色の数

    Returns:
    -------
        カラーマップの配列

    """
    return plt.get_cmap("viridis")(np.linspace(0, 1, n_colors))


def setup_subplots(
    rows: int, cols: int, figsize: tuple[int, int]
) -> tuple[Figure, list[Axes]]:
    """サブプロットの初期設定.

    Args:
    ----
        rows: 行数
        cols: 列数
        figsize: 図のサイズ

    Returns:
    -------
        図とAxesのリスト

    """
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    return fig, list(axes.flatten())  # ndarrayをlistに変換
