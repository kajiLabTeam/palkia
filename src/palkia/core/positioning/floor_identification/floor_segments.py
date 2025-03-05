# palkia/positioning/floor_identification/floor_segments.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from palkia.core.map.floor_map import FloorMap


@dataclass
class FloorInfo:
    """Floor information container.

    Attributes
    ----------
        floor_number: The floor number (1F = 1, B1F = -1).
        pressure_range: The pressure range for this floor (min, max).
        time_intervals: List of time intervals spent on this floor.
        trajectory: Trajectory data for this floor.
        floor_map: Optional floor map associated with this floor.

    """

    floor_number: int
    pressure_range: tuple[float, float]
    time_intervals: list[tuple[float, float]]
    trajectory: pd.DataFrame
    floor_map: FloorMap | None = None
