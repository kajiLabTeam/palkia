from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .plot_utils import setup_axis

if TYPE_CHECKING:
    import pandas as pd

    from palkia.positioning.floor_identification import FloorInfo


def plot_pressure_analysis(
    baro_data: pd.DataFrame,
    floor_info: dict[int, FloorInfo],
    transitions: list[dict],
) -> None:
    """気圧データと階層推定の分析結果をプロット.

    Args:
    ----
        baro_data: 気圧センサーデータ
        floor_info: 階層ごとの情報
        transitions: 階層間の移動情報

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # 気圧データ
    ax1.plot(baro_data["ts"], baro_data["pressure"], "b-", alpha=0.5, label="Raw")
    ax1.plot(baro_data["ts"], baro_data["pressure_smoothed"], "r-", label="Smoothed")

    # 階層区分
    for floor, info in floor_info.items():
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
    for floor, info in floor_info.items():
        for start, end in info.time_intervals:
            floor_numbers.extend([floor, floor])
            times.extend([start, end])

    ax2.step(times, floor_numbers, where="post")
    setup_axis(ax2, "", xlabel="Time (s)", ylabel="Floor Number", aspect=None)

    plt.tight_layout()
    plt.show()
