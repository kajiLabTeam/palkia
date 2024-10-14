from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class MapMatcher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.map_data = self._load_map_data(config.get("map_file"))
        self.grid_size = config.get("grid_size", 0.5)  # メートル単位のグリッドサイズ
        self.max_search_distance = config.get(
            "max_search_distance", 5
        )  # 最大探索距離（メートル）
        self.kdtree = self._build_kdtree()

    def _load_map_data(self, map_file: str) -> np.ndarray:
        """地図データを読み込む"""
        # 注: 実際の実装では、適切な地図データフォーマットに合わせて読み込みロジックを実装する必要があります
        return np.load(map_file)

    def _build_kdtree(self) -> cKDTree:
        """歩行可能な点のKDツリーを構築する"""
        walkable_points = np.argwhere(self.map_data == 1)  # 1を歩行可能とする
        return cKDTree(walkable_points * self.grid_size)

    def match(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """軌跡を地図にマッチングする

        Args:
            trajectory (pd.DataFrame): マッチングする軌跡

        Returns:
            pd.DataFrame: マッチングされた軌跡

        """
        matched_trajectory = trajectory.copy()
        for i, point in trajectory.iterrows():
            matched_point = self._match_point(point["x"], point["y"])
            matched_trajectory.at[i, "x"] = matched_point[0]
            matched_trajectory.at[i, "y"] = matched_point[1]
        return matched_trajectory

    def _match_point(self, x: float, y: float) -> Tuple[float, float]:
        """単一の点を最も近い歩行可能な点にマッチングする"""
        distance, index = self.kdtree.query(
            [x, y], distance_upper_bound=self.max_search_distance
        )
        if np.isinf(distance):
            return x, y  # マッチする点が見つからない場合は元の点を返す
        matched_point = self.kdtree.data[index] * self.grid_size
        return matched_point[0], matched_point[1]

    def correct_trajectory(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        """軌跡を補正する

        Args:
            trajectory (pd.DataFrame): 補正する軌跡

        Returns:
            pd.DataFrame: 補正された軌跡

        """
        corrected_trajectory = trajectory.copy()
        for i in range(len(corrected_trajectory)):
            if not self.is_walkable(
                corrected_trajectory.at[i, "x"], corrected_trajectory.at[i, "y"]
            ):
                nearest_walkable = self._find_nearest_walkable(
                    corrected_trajectory.at[i, "x"], corrected_trajectory.at[i, "y"]
                )
                corrected_trajectory.at[i, "x"] = nearest_walkable[0]
                corrected_trajectory.at[i, "y"] = nearest_walkable[1]
        return corrected_trajectory

    def is_walkable(self, x: float, y: float) -> bool:
        """指定された座標が歩行可能かどうかを判定する"""
        grid_x, grid_y = int(x / self.grid_size), int(y / self.grid_size)
        if (
            0 <= grid_x < self.map_data.shape[0]
            and 0 <= grid_y < self.map_data.shape[1]
        ):
            return self.map_data[grid_x, grid_y] == 1
        return False

    def _find_nearest_walkable(self, x: float, y: float) -> Tuple[float, float]:
        """最も近い歩行可能な点を見つける"""
        distance, index = self.kdtree.query([x, y])
        nearest_point = self.kdtree.data[index] * self.grid_size
        return nearest_point[0], nearest_point[1]

    def smooth_trajectory(
        self, trajectory: pd.DataFrame, window_size: int = 5
    ) -> pd.DataFrame:
        """軌跡を平滑化する

        Args:
            trajectory (pd.DataFrame): 平滑化する軌跡
            window_size (int): 移動平均のウィンドウサイズ

        Returns:
            pd.DataFrame: 平滑化された軌跡

        """
        smoothed = trajectory.copy()
        smoothed[["x", "y"]] = (
            smoothed[["x", "y"]]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        return smoothed
