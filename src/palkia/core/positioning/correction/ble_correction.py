from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class BLECorrector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.beacon_positions = config.get(
            "beacon_positions", {}
        )  # BLEビーコンの位置情報を読み込む(設定ファイルから)
        self.rssi_threshold = config.get("rssi_threshold", -80)  # RSSIの閾値
        self.time_window = config.get("time_window", 5)  # マッチングの時間窓(秒)

    def correct(self, trajectory: pd.DataFrame, ble_data: pd.DataFrame) -> pd.DataFrame:
        """BLEデータを使用して軌跡を補正する.

        Args:
            trajectory (pd.DataFrame): 補正する軌跡
            ble_data (pd.DataFrame): BLEスキャンデータ

        Returns:
            pd.DataFrame: BLE補正された軌跡

        """
        # 強いRSSI値のBLEスキャンのみをフィルタリング
        strong_ble_scans = self._filter_strong_blescans(ble_data)

        # 補正された軌跡を初期化
        corrected_trajectory = trajectory.copy()

        # 時間窓ごとに処理
        for start_time in np.arange(
            trajectory["ts"].min(), trajectory["ts"].max(), self.time_window
        ):
            end_time = start_time + self.time_window

            # 時間窓内の軌跡とBLEスキャンを抽出
            window_trajectory = trajectory[
                (trajectory["ts"] >= start_time) & (trajectory["ts"] < end_time)
            ]
            window_ble_scans = strong_ble_scans[
                (strong_ble_scans["ts"] >= start_time)
                & (strong_ble_scans["ts"] < end_time)
            ]

            if not window_ble_scans.empty:
                # BLEデータを使用して最適な位置を推定
                optimal_position = self._estimate_position_from_ble(window_ble_scans)

                # 軌跡を補正
                corrected_trajectory.loc[
                    (corrected_trajectory["ts"] >= start_time)
                    & (corrected_trajectory["ts"] < end_time),
                    ["x", "y"],
                ] = self._adjust_trajectory(
                    window_trajectory[["x", "y"]], optimal_position
                )

        return corrected_trajectory

    def _filter_strong_blescans(self, ble_data: pd.DataFrame) -> pd.DataFrame:
        """強いRSSI値のBLEスキャンのみをフィルタリングする."""
        return ble_data[ble_data["rssi"] > self.rssi_threshold].copy()

    def _estimate_position_from_ble(
        self, ble_scans: pd.DataFrame
    ) -> dict[str, float] | None:
        """BLEスキャンデータから位置を推定する."""
        weighted_positions = []
        weights = []

        for _, scan in ble_scans.iterrows():
            if scan["bdaddress"] in self.beacon_positions:
                beacon_pos = self.beacon_positions[scan["bdaddress"]]
                # RSSIの値が大きいほど重みを大きくする
                weight = 10 ** (scan["rssi"] / 10)  # 例: -70dBm → 0.1, -60dBm → 0.25
                weighted_positions.append(
                    [beacon_pos["x"] * weight, beacon_pos["y"] * weight]
                )
                weights.append(weight)

        if weighted_positions:
            # 重み付き平均を計算
            estimated_position = np.average(weighted_positions, axis=0, weights=weights)
            return {"x": estimated_position[0], "y": estimated_position[1]}
        return None

    def _adjust_trajectory(
        self,
        trajectory_segment: pd.DataFrame,
        optimal_position: dict[str, float] | None,
    ) -> pd.DataFrame:
        """軌跡セグメントを最適位置に合わせて調整する."""
        if optimal_position is None:
            return trajectory_segment

        # 軌跡セグメントの中心を計算
        center = trajectory_segment.mean()

        # 移動量を計算
        delta_x = optimal_position["x"] - center["x"]
        delta_y = optimal_position["y"] - center["y"]

        # 軌跡を移動
        adjusted_segment = trajectory_segment.copy()
        adjusted_segment["x"] += delta_x
        adjusted_segment["y"] += delta_y

        return adjusted_segment

    def _calculate_rssi_to_distance(self, rssi: float) -> float:
        """RSSI値を距離に変換する(簡易的なモデル)."""
        # 参考: https://iotandelectronics.wordpress.com/2016/10/07/how-to-calculate-distance-from-the-rssi-value-of-the-ble-beacon/
        # 1m距離での理想的なRSSI値(デバイスによって異なる)
        tx_power = -59
        ratio = rssi * 1.0 / tx_power
        if ratio < 1.0:
            return ratio**10
        return 0.89976 * (ratio**7.7095) + 0.111
