from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from palkia.config.column_name import PRESSURE, TIMESTAMP

from .plot_utils import setup_axis

if TYPE_CHECKING:
    import pandas as pd

    from palkia.core.positioning.floor_identification import FloorInfo

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


def plot_pressure_analysis(
    baro_data: pd.DataFrame,
    floor_segments: dict[int, FloorInfo],
    transitions: list[dict],
) -> None:
    """気圧データと階層推定の分析結果をプロット.

    Args:
    ----
        baro_data: 気圧センサーデータ
        floor_segments: 階層ごとの情報
        transitions: 階層間の移動情報

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # 気圧データ
    ax1.plot(baro_data["ts"], baro_data["pressure"], "b-", alpha=0.5, label="Raw")
    ax1.plot(baro_data["ts"], baro_data["pressure_smoothed"], "r-", label="Smoothed")

    # 階層区分
    for floor, info in floor_segments.items():
        p_min, p_max = info.pressure_range
        ax1.axhspan(p_min, p_max, alpha=0.2, label=f"Floor {floor}")

    # 移動区間
    for t in transitions:
        ax1.axvspan(
            t["start_time"],
            t["end_time"],
            color="y",
            alpha=0.3,
        )

    setup_axis(ax1, "", ylabel="Pressure (hPa)", aspect=None)
    ax1.legend()

    # 階層推定結果
    floor_numbers = []
    times = []
    for floor, info in floor_segments.items():
        for start, end in info.time_intervals:
            floor_numbers.extend([floor, floor])
            times.extend([start, end])

    ax2.step(times, floor_numbers, where="post")
    setup_axis(ax2, "", xlabel="Time (s)", ylabel="Floor Number", aspect=None)

    plt.tight_layout()
    plt.show()


def plot_pressure_with_stable_regions(
    baro_data: pd.DataFrame,
    stable_intervals: list[tuple[float, float]],
    pressure_threshold: float,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """気圧データと安定歩行区間を可視化.

    Args:
        baro_data: 気圧センサーデータ
        stable_intervals: 安定区間のリスト [(start_time, end_time), ...]
        pressure_threshold: 安定判定の閾値
        figsize: 図のサイズ

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # 気圧データの表示
    ax1.plot(
        baro_data[TIMESTAMP], baro_data[PRESSURE], "b-", alpha=0.6, label="Raw Pressure"
    )


    for start, end in stable_intervals:
        ax1.axvspan(start, end, color="g", alpha=0.2)

        # 該当区間の平均気圧を表示
        interval_data = baro_data[
            (baro_data[TIMESTAMP] >= start) & (baro_data[TIMESTAMP] <= end)
        ]
        mean_pressure = interval_data[PRESSURE].mean()
        ax1.axhspan(
            mean_pressure - pressure_threshold,
            mean_pressure + pressure_threshold,
            xmin=(start - baro_data[TIMESTAMP].min())
            / (baro_data[TIMESTAMP].max() - baro_data[TIMESTAMP].min()),
            xmax=(end - baro_data[TIMESTAMP].min())
            / (baro_data[TIMESTAMP].max() - baro_data[TIMESTAMP].min()),
            color="y",
            alpha=0.2,
        )

    setup_axis(
        ax1,
        "Barometer Data and Stable Regions",
        xlabel="Time (s)",
        ylabel="Pressure (hPa)",
        aspect=None,
    )
    ax1.legend()

    # 安定度の可視化
    pressure_diff = baro_data[PRESSURE].diff().abs()
    stability = 1.0 / (1.0 + pressure_diff)
    ax2.plot(baro_data[TIMESTAMP], stability, "g-", alpha=0.6, label="Stability")

    # 安定区間のマーカー
    stability_threshold = 1.0 / (1.0 + pressure_threshold)
    ax2.axhline(
        y=stability_threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label="Stability Threshold",
    )

    setup_axis(
        ax2, "Stability Index", xlabel="Time (s)", ylabel="Stability", aspect=None
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_floor_transitions(
    baro_data: pd.DataFrame,
    stable_intervals: list[tuple[float, float]],
    pressure_levels: dict[int, float],
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """階層変化を可視化.

    Args:
        baro_data: 気圧センサーデータ
        stable_intervals: 安定区間のリスト
        pressure_levels: 階層ごとの基準気圧
        figsize: 図のサイズ

    """
    fig, ax = plt.subplots(figsize=figsize)

    # 気圧データ
    ax.plot(baro_data[TIMESTAMP], baro_data[PRESSURE], "b-", alpha=0.6, label="気圧")

    # 階層の基準気圧を表示
    for floor, pressure in pressure_levels.items():
        ax.axhline(
            y=pressure,
            color=f"C{floor+1+3}",
            linestyle="--",
            alpha=0.5,
            label=f"フロア {floor+1+3}階",
        )

    # 安定区間
    for start, end in stable_intervals:
        ax.axvspan(start, end, color="g", alpha=0.2)

    setup_axis(
        ax, "Floor Transitions", xlabel="時間 (s)", ylabel="気圧 (hPa)", aspect=None
    )
    ax.legend()

    plt.tight_layout()
    plt.show()
