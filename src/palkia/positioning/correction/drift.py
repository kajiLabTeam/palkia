from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from palkia.positioning.pdr import (
    OrientationEstimator,
    PDREstimator,
    StepEstimator,
    TrajectoryCalculator,
)


class DriftCorrector:
    def __init__(self, config: Dict[str, Any], pdr_estimator: PDREstimator) -> None:
        self.config = config
        # ドリフト探索の範囲とステップ幅を設定
        self.drift_search_range = config.get("drift_search_range", (-0.05, 0.05))
        self.drift_search_step = config.get("drift_search_step", 0.001)
        self.pdr_estimator = pdr_estimator

    def correct(
        self, trajectory: pd.DataFrame, gyro_df: pd.DataFrame, acc_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ジャイロと加速度データを使用して軌跡のドリフトを補正する。.

        Args:
            trajectory (pd.DataFrame): 補正する軌跡
            gyro_df (pd.DataFrame): ジャイロデータ
            acc_df (pd.DataFrame): 加速度データ

        Returns:
            pd.DataFrame: ドリフト補正された軌跡

        """
        # ジャイロデータを角度データに変換
        angle_df = self._convert_gyro_to_angle(gyro_df)
        # 最適なドリフト値を探索
        optimal_drift = self._search_optimal_drift(angle_df, acc_df, trajectory)
        # 最適なドリフト値を使用して角度データを補正
        corrected_angle_df = self._apply_drift_correction(angle_df, optimal_drift)
        # 補正された角度データを使用して軌跡を更新
        return self.pdr_estimator.estimate_trajectory(trajectory, corrected_angle_df)

    def _convert_gyro_to_angle(self, gyro_df: pd.DataFrame) -> pd.DataFrame:
        """ジャイロデータを角度データに変換する."""
        angle_df = pd.DataFrame()
        angle_df["ts"] = gyro_df["ts"]
        # サンプリングレートが100Hzであると仮定
        angle_df["x"] = gyro_df["x"].cumsum() * 0.01
        return angle_df

    def _search_optimal_drift(
        self, angle_df: pd.DataFrame, acc_df: pd.DataFrame, trajectory: pd.DataFrame
    ) -> float:
        """最適なドリフト値を探索する."""
        min_drift, max_drift = self.drift_search_range
        drift_range = np.arange(min_drift, max_drift, self.drift_search_step)

        best_drift = 0
        min_error = float("inf")

        for drift in drift_range:
            # 各ドリフト値に対して角度データを補正
            corrected_angle = self._apply_drift_correction(angle_df, drift)
            # 補正後の誤差を計算
            error = self._calculate_error(corrected_angle, acc_df, trajectory)
            if error < min_error:
                min_error = error
                best_drift = drift

        return best_drift

    def _apply_drift_correction(
        self, angle_df: pd.DataFrame, drift: float
    ) -> pd.DataFrame:
        """角度データにドリフト補正を適用する."""
        corrected_angle = angle_df.copy()
        base_time = corrected_angle["ts"].iloc[0]
        elapsed_time = corrected_angle["ts"] - base_time
        corrected_angle["x"] -= drift * elapsed_time
        return corrected_angle

    def _calculate_error(
        self, angle_df: pd.DataFrame, acc_df: pd.DataFrame, trajectory: pd.DataFrame
    ) -> float:
        """補正された軌跡と元の軌跡との誤差を計算する."""
        corrected_trajectory = self._update_trajectory_with_corrected_angle(
            trajectory, angle_df
        )
        # 加速度のピークを検出（歩行のタイミングを推定）
        peaks, _ = find_peaks(acc_df["norm"], height=12, distance=10)
        peak_times = acc_df.iloc[peaks]["ts"].values

        # ピーク時の位置を比較
        original_positions = trajectory[trajectory["ts"].isin(peak_times)][
            ["x", "y"]
        ].values
        corrected_positions = corrected_trajectory[
            corrected_trajectory["ts"].isin(peak_times)
        ][["x", "y"]].values

        # 平均二乗誤差の平方根（RMSE）を計算
        error = np.mean(
            np.sqrt(np.sum((original_positions - corrected_positions) ** 2, axis=1))
        )
        return error

    def _update_trajectory_with_corrected_angle(
        self, trajectory: pd.DataFrame, corrected_angle: pd.DataFrame
    ) -> pd.DataFrame:
        """補正された角度データを使用して軌跡を更新する."""
        corrected_trajectory = trajectory.copy()
        angle_interpolator = self._create_angle_interpolator(corrected_angle)

        # デフォルトの歩幅を設定（設定ファイルから読み込む）
        step_length = self.config.get("step_length", 0.5)
        cumulative_x = 0
        cumulative_y = 0

        for i, row in corrected_trajectory.iterrows():
            # 各時点での角度を補間
            angle = angle_interpolator(row["ts"])
            # 累積変位を計算
            cumulative_x += step_length * np.cos(angle)
            cumulative_y += step_length * np.sin(angle)
            corrected_trajectory.at[i, "x"] = cumulative_x
            corrected_trajectory.at[i, "y"] = cumulative_y

        return corrected_trajectory

    def _create_angle_interpolator(self, angle_df: pd.DataFrame):
        """角度データの補間関数を作成する."""
        from scipy.interpolate import interp1d

        return interp1d(
            angle_df["ts"],
            angle_df["x"],
            kind="linear",
            fill_value="extrapolate",  # type: ignore
            bounds_error=False,
        )
