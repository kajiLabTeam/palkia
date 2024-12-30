from typing import TYPE_CHECKING

import pandas as pd

from palkia.core.sensor_processing.enhanced_sensor_data import EnhancedSensorData

if TYPE_CHECKING:
    from palkia.core.map.floor_map import FloorMap


class PDRSession:
    def __init__(self, enhanced_sensor_data: EnhancedSensorData) -> None:
        self.sensor_data = enhanced_sensor_data
        self.raw_trajectory: pd.DataFrame | None = None
        self.current_trajectory: pd.DataFrame | None = None
        self.floor_info: dict[int, FloorMap] | None = None

    def update_trajectory(self, new_trajectory: pd.DataFrame) -> None:
        """補正済み軌跡を更新."""
        if self.raw_trajectory is None:
            self.raw_trajectory = new_trajectory.copy()
        self.current_trajectory = new_trajectory
