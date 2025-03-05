from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from palkia.config.column_name import COORDINATE_X, COORDINATE_Y, TRANSMITTER_ID


@dataclass
class Point2D:
    x: float
    y: float


class WirelessSignalCorrector:
    def __init__(
        self,
        signal_realtime_scans: pd.DataFrame,
        signal_fingerprints: pd.DataFrame | None = None,
        transmitter_positions: pd.DataFrame | None = None,
        rssi_threshold: int = -70,
        time_window: int = 5,
    ) -> None:
        self.signal_realtime_scans = signal_realtime_scans
        self.signal_fingerprints = signal_fingerprints
        self.transmitter_positions = transmitter_positions
        self.rssi_threshold = rssi_threshold
        self.time_window = time_window

    def correct_initial_direction_with_transmitter_positions(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:
        """データを使用して軌跡を補正する.

        Args:
        ----
            trajectory: 補正する軌跡

        Returns:
        -------
            信号補正された軌跡

        """
        if self.transmitter_positions is None:
            msg = "The attribute 'transmitter_positions' is None."
            raise ValueError(msg)

        strong_signal_scans = self._filter_strong_signal_scans(self.signal_realtime_scans)
        strong_signal_merged = strong_signal_scans.merge(
            self.transmitter_positions, on=TRANSMITTER_ID, how="left"
        ).rename(columns={"x": "transmitter_x", "y": "transmitter_y"})

        initial_point = Point2D(
            x=trajectory[COORDINATE_X].iloc[0], y=trajectory[COORDINATE_Y].iloc[0]
        )

        return self._optimize_trajectory_rotation(
            trajectory.copy(), strong_signal_merged, initial_point
        )

    def correct_initial_direction_with_fp(
        self,
        trajectory: pd.DataFrame,
    ) -> pd.DataFrame:
        """FPデータを使用して軌跡を補正."""
        if self.signal_fingerprints is None:
            msg = "The attribute 'signal_fingerprints' is None."
            raise ValueError(msg)

        strong_signal_scans = self._filter_strong_signal_scans(self.signal_realtime_scans)
        # 処理に時間がかかるため注意が必要
        strong_signal_merged = self._estimate_positions_from_fp(
            strong_signal_scans, self.signal_fingerprints
        )

        initial_point = Point2D(x=trajectory["x"].iloc[0], y=trajectory["y"].iloc[0])

        return self._optimize_trajectory_rotation(
            trajectory.copy(), strong_signal_merged, initial_point
        )

    def _calculate_rssi_weight(
        self,
        rssi_value: float,
        rssi_mean: float,
        rssi_std: float,
        min_std: float = 1.0,  # 最小標準偏差
        min_weight: float = 1e-10,  # 最小重み
    ) -> float:
        """RSSIの値に基づいて重みを計算.

        Args:
            rssi_value: 現在のRSSI値
            rssi_mean: FPデータのRSSI平均値
            rssi_std: FPデータのRSSI標準偏差
            min_std: 最小標準偏差(ゼロ除算防止用)
            min_weight: 最小重み

        Returns:
            float: 計算された重み

        """
        # 標準偏差が0またはNaNの場合、min_stdを使用
        std = max(rssi_std if not np.isnan(rssi_std) else 0, min_std)

        # 重みの計算
        weight = np.exp(-0.5 * ((rssi_value - rssi_mean) / std) ** 2)

        # 最小重みを保証
        return max(weight, min_weight)

    def _calculate_rssi_weight_with_path_loss(
        self,
        rssi_value: float,
        reference_rssi: float,
        path_loss_exponent: float = 2.0,
        reference_distance: float = 1.0,
        min_weight: float = 1e-10,
    ) -> float:
        """RSSIの値に基づいて重みを計算(パスロスモデル使用).

        Args:
            rssi_value: 現在のRSSI値
            reference_rssi: 基準距離での参照RSSI値
            path_loss_exponent: パスロス指数(環境による、通常2-4の範囲)
            reference_distance: 基準距離(メートル)
            min_weight: 最小重み

        Returns:
            float: 計算された重み

        """
        # RSSIから距離を推定(対数距離損失モデル)
        estimated_distance = reference_distance * 10 ** (
            (reference_rssi - rssi_value) / (10 * path_loss_exponent)
        )

        # 距離の逆数を重みとして使用(距離が遠いほど重みが小さくなる)
        weight = 1 / (estimated_distance**2)

        return max(weight, min_weight)

    def _calculate_hybrid_weight(
        self,
        rssi_value: float,
        rssi_mean: float,
        rssi_std: float,
        reference_rssi: float = -76,
        path_loss_exponent: float = 2.0,
        alpha: float = 0.5,  # ブレンド係数
        min_weight: float = 1e-10,
    ) -> float:
        """ガウシアンモデルとパスロスモデルを組み合わせた重み計算.

        Args:
            rssi_value: 現在のRSSI値
            rssi_mean: FPデータのRSSI平均値
            rssi_std: FPデータのRSSI標準偏差
            reference_rssi: 基準距離での参照RSSI値
            path_loss_exponent: パスロス指数
            alpha: ブレンド係数(0-1)、1に近いほどガウシアンモデルの影響が強くなる
            min_weight: 最小重み

        Returns:
            float: 計算された重み

        """
        gaussian_weight = self._calculate_rssi_weight(rssi_value, rssi_mean, rssi_std)
        path_loss_weight = self._calculate_rssi_weight_with_path_loss(
            rssi_value, reference_rssi, path_loss_exponent
        )

        # 重みの線形結合
        combined_weight = alpha * gaussian_weight + (1 - alpha) * path_loss_weight

        return max(combined_weight, min_weight)

    def _get_alpha(self, sample_count: int, min_samples: int = 3) -> float:
        if sample_count < min_samples:
            return 0.2  # パスロスモデルの重みを大きくする
        return 0.7

    def _estimate_beacon_position(
        self, fp_data: pd.DataFrame, beacon_address: str, target_rssi: float
    ) -> tuple[float, float]:
        """基地局の位置を推定."""
        beacon_fp = fp_data[fp_data["beacon_address"] == beacon_address]

        if beacon_fp.empty:
            print(f"Warning: No FP data found for beacon {beacon_address}")
            return 0.0, 0.0  # もしくは適切なデフォルト値

        weights = np.array(
            [
                self._calculate_hybrid_weight(
                    row.loc["rssi_mean"],
                    target_rssi,
                    row.loc["rssi_std"],
                    self._get_alpha(row.loc["count"]),
                )
                for _, row in beacon_fp.iterrows()
            ]
        )

        estimated_x = np.average(beacon_fp["x"], weights=weights)
        estimated_y = np.average(beacon_fp["y"], weights=weights)

        return estimated_x, estimated_y

    def _rotate_trajectory(
        self, df: pd.DataFrame, angle: float, initial_point: Point2D
    ) -> pd.DataFrame:
        x_displacement = df[COORDINATE_X] - initial_point.x
        y_displacement = df[COORDINATE_Y] - initial_point.y

        rotated_x = (
            x_displacement * np.cos(angle)
            - y_displacement * np.sin(angle)
            + initial_point.x
        )
        rotated_y = (
            x_displacement * np.sin(angle)
            + y_displacement * np.cos(angle)
            + initial_point.y
        )

        return pd.DataFrame(
            {"ts": df.ts, COORDINATE_X: rotated_x, COORDINATE_Y: rotated_y}
        )

    def _estimate_positions_from_fp(
        self, signal_realtime_scans: pd.DataFrame, signal_fingerprints: pd.DataFrame
    ) -> pd.DataFrame:
        """全ての信号データに対して位置を推定."""
        result_data = signal_realtime_scans.copy()
        result_data["transmitter_x"] = 0.0
        result_data["transmitter_y"] = 0.0

        for idx, row in result_data.iterrows():
            x, y = self._estimate_beacon_position(
                signal_fingerprints, row.loc[TRANSMITTER_ID], row.loc["rssi"]
            )
            result_data.loc[idx, "transmitter_x"] = x
            result_data.loc[idx, "transmitter_y"] = y

        return result_data

    def _calculate_total_distance(
        self, trajectory: pd.DataFrame, signal_realtime_scans: pd.DataFrame
    ) -> float:
        merged = pd.merge_asof(
            trajectory.sort_values("ts"),
            signal_realtime_scans.sort_values("ts"),
            on="ts",
            direction="nearest",
        )

        return np.sqrt(
            (merged[COORDINATE_X] - merged["transmitter_x"]) ** 2
            + (merged[COORDINATE_Y] - merged["transmitter_y"]) ** 2
        ).sum()

    def _optimize_trajectory_rotation(
        self,
        trajectory: pd.DataFrame,
        signal_realtime_scans: pd.DataFrame,
        initial_point: Point2D,
    ) -> pd.DataFrame:
        angles = np.arange(0, 2 * np.pi, 0.01)
        optimal_angle = min(
            angles,
            key=lambda angle: self._calculate_total_distance(
                self._rotate_trajectory(trajectory, angle, initial_point),
                signal_realtime_scans,
            ),
        )

        return self._rotate_trajectory(trajectory, optimal_angle, initial_point)

    def _filter_strong_signal_scans(
        self, signal_realtime_scans: pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        """強いRSSI値の信号スキャンのみをフィルタリングする."""
        return signal_realtime_scans[
            signal_realtime_scans["rssi"] > self.rssi_threshold
        ].copy()
