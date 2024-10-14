from typing import Any, Dict

import numpy as np
import pandas as pd

from palkia.positioning.pdr import (
    OrientationEstimator,
    PDREstimator,
    StepEstimator,
    TrajectoryCalculator,
)

from .ble_correction import BLECorrector
from .drift import DriftCorrector
from .map_matching import MapMatcher
from .stable_walking import StableWalkingDetector


class TrajectoryCorrector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pdr_estimator = PDREstimator(
            StepEstimator(config.get("step_estimator", {})),
            OrientationEstimator(config.get("orientation_estimator", {})),
            TrajectoryCalculator(
                config.get("trajectory_calculator", {"initial_point": {"x": 0, "y": 0}})
            ),
        )
        self.drift_corrector = DriftCorrector(
            config.get("drift", {}), self.pdr_estimator
        )
        self.map_matcher = MapMatcher(config.get("map_matching", {}))
        self.ble_corrector = BLECorrector(config.get("ble", {}))
        self.stable_walking_detector = StableWalkingDetector(
            config.get("stable_walking", {})
        )
        self.initial_point = config.get("initial_point", {"x": 0, "y": 0})

    def estimate_and_correct_trajectory(
        self,
        acc_data: pd.DataFrame,
        gyro_data: pd.DataFrame,
        ble_data: pd.DataFrame | None = None,
        floor_map: Any = None,
    ) -> pd.DataFrame:
        """PDRを用いて軌跡を推定し、各種補正を適用する.

        Args:
            acc_data (pd.DataFrame): 加速度データ
            gyro_data (pd.DataFrame): ジャイロデータ
            ble_data (pd.DataFrame, optional): BLEデータ
            floor_map (Any, optional): フロアマップデータ

        Returns:
            pd.DataFrame: 推定・補正された軌跡

        """
        # 初期軌跡の計算
        trajectory = self.pdr_estimator.estimate_trajectory(acc_data, gyro_data)
        # ドリフト補正
        trajectory = self.drift_corrector.correct(trajectory, gyro_data, acc_data)

        # マップマッチング
        # if floor_map is not None:
        #     trajectory = self.map_matcher.match(trajectory, floor_map)

        # BLE補正
        # if ble_data is not None:
        #     trajectory = self.ble_corrector.correct(trajectory, ble_data)

        # 安定歩行検出と補正
        # trajectory = self.stable_walking_detector.detect_and_correct(
        #     trajectory, acc_data
        # )

        return trajectory

    def analyze_corrections(
        self, original_trajectory: pd.DataFrame, corrected_trajectory: pd.DataFrame
    ) -> Dict[str, Any]:
        """補正前後の軌跡の差異を分析する"""
        analysis = {}

        # 総移動距離の計算
        original_distance = self._calculate_total_distance(original_trajectory)
        corrected_distance = self._calculate_total_distance(corrected_trajectory)
        analysis["distance_change"] = corrected_distance - original_distance

        # 最大偏差の計算
        max_deviation = self._calculate_max_deviation(
            original_trajectory, corrected_trajectory
        )
        analysis["max_deviation"] = max_deviation

        return analysis

    def _calculate_total_distance(self, trajectory: pd.DataFrame) -> float:
        """軌跡の総移動距離を計算する"""
        distances = np.sqrt(
            np.diff(trajectory["x"]) ** 2 + np.diff(trajectory["y"]) ** 2
        )
        return np.sum(distances)

    def _calculate_max_deviation(
        self, trajectory1: pd.DataFrame, trajectory2: pd.DataFrame
    ) -> float:
        """2つの軌跡間の最大偏差を計算する"""
        deviations = np.sqrt(
            (trajectory1["x"] - trajectory2["x"]) ** 2
            + (trajectory1["y"] - trajectory2["y"]) ** 2
        )
        return np.max(deviations)

    def visualize_trajectory(
        self,
        trajectory: pd.DataFrame,
        floor_map: Any = None,
        ground_truth: pd.DataFrame = None,
    ):
        """軌跡を可視化する"""
        # 実装は省略（matplotlib や plotly などを使用して可視化）
