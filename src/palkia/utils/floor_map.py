import numpy as np
from PIL import Image


class FloorMap:
    def __init__(
        self,
        floor_name: str,
        floor_map_path: str,
        dx: float,
        dy: float,
    ) -> None:
        self.floor_name = floor_name
        self.floor_map_data = FloorMap.__load_floor_map(floor_map_path)
        # ピクセル間の距離
        self.dx = dx
        self.dy = dy

    @staticmethod
    def __load_floor_map(
        floor_map_path: str,
    ) -> np.ndarray:
        map_image_path = floor_map_path
        map_image = Image.open(map_image_path)

        return np.array(map_image, dtype=bool)
