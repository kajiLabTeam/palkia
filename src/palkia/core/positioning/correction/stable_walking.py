from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class StableWalkingDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stable_angle_range = config.get("stable_angle_range", 0.1)  # ラジアン
        self.stable_time = config.get("stable_time", 3.0)  # 秒
        self.step_length = config.get("step_length", 0.5)  # メートル

    def detect_and_correct(
        self, trajectory: pd.DataFrame, acc_data: pd.DataFrame
    ) -> pd.DataFrame:
        """安定歩行区間を検出し、軌跡を補正する

        Args:
            trajectory (pd.DataFrame): 補正する軌跡
            acc_data (pd.DataFrame): 加速度データ

        Returns:
            pd.DataFrame: 補正された軌跡

        """
        # 角度データを計算
        angle_data = self._calculate_angles(trajectory)

        # 安定歩行区間を検出
        stable_segments = self._detect_stable_segments(angle_data)

        # 歩行のピークを検出
        steps = self._detect_steps(acc_data)

        # 補正された軌跡を初期化
        corrected_trajectory = trajectory.copy()

        for segment in stable_segments:
            start_time, end_time = segment
            segment_steps = steps[(steps >= start_time) & (steps <= end_time)]

            if len(segment_steps) > 1:
                # 安定歩行区間内の軌跡を補正
                corrected_segment = self._correct_segment(
                    trajectory[
                        (trajectory["ts"] >= start_time)
                        & (trajectory["ts"] <= end_time)
                    ],
                    segment_steps,
                )
                corrected_trajectory.loc[
                    (corrected_trajectory["ts"] >= start_time)
                    & (corrected_trajectory["ts"] <= end_time)
                ] = corrected_segment

        return corrected_trajectory

    def _calculate_angles(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """軌跡から角度を計算する"""
        dx = trajectory["x"].diff()
        dy = trajectory["y"].diff()
        angles = np.arctan2(dy, dx)
        return pd.DataFrame({"ts": trajectory["ts"], "angle": angles})

    def _detect_stable_segments(
        self, angle_data: pd.DataFrame
    ) -> List[Tuple[float, float]]:
        """安定歩行区間を検出する"""
        stable_segments = []
        start_time = None
        prev_angle = None

        for _, row in angle_data.iterrows():
            if prev_angle is None:
                prev_angle = row["angle"]
                start_time = row["ts"]
                continue

            if abs(row["angle"] - prev_angle) <= self.stable_angle_range:
                if row["ts"] - start_time >= self.stable_time:
                    stable_segments.append((start_time, row["ts"]))
            else:
                start_time = row["ts"]

            prev_angle = row["angle"]

        return stable_segments

    def _detect_steps(self, acc_data: pd.DataFrame) -> np.ndarray:
        """加速度データから歩行のピークを検出する"""
        # 加速度の大きさを計算
        acc_magnitude = np.sqrt(
            acc_data["x"] ** 2 + acc_data["y"] ** 2 + acc_data["z"] ** 2
        )

        # ピーク検出
        peaks, _ = find_peaks(
            acc_magnitude, height=10, distance=50
        )  # パラメータは調整が必要

        return acc_data["ts"].iloc[peaks].values

    def _correct_segment(
        self, segment: pd.DataFrame, steps: np.ndarray
    ) -> pd.DataFrame:
        """安定歩行区間内の軌跡を補正する"""
        corrected_segment = segment.copy()

        start_point = segment.iloc[0][["x", "y"]].values
        end_point = segment.iloc[-1][["x", "y"]].values

        # 歩数から予想される距離を計算
        expected_distance = len(steps) * self.step_length

        # 実際の距離を計算
        actual_distance = np.linalg.norm(end_point - start_point)

        # スケール係数を計算
        scale_factor = expected_distance / actual_distance if actual_distance > 0 else 1

        # 軌跡をスケーリング
        corrected_segment["x"] = (
            start_point[0] + (corrected_segment["x"] - start_point[0]) * scale_factor
        )
        corrected_segment["y"] = (
            start_point[1] + (corrected_segment["y"] - start_point[1]) * scale_factor
        )

        return corrected_segment

    def visualize_stable_segments(
        self, trajectory: pd.DataFrame, stable_segments: List[Tuple[float, float]]
    ):
        """安定歩行区間を可視化する"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(trajectory["x"], trajectory["y"], label="Original Trajectory")

        for start, end in stable_segments:
            segment = trajectory[
                (trajectory["ts"] >= start) & (trajectory["ts"] <= end)
            ]
            plt.plot(segment["x"], segment["y"], "r", linewidth=2)

        plt.legend()
        plt.title("Trajectory with Stable Walking Segments")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
