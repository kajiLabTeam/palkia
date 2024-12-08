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
        # ピクセル間の距離(m)  # noqa: ERA001
        self.dx = dx
        self.dy = dy

    @staticmethod
    def __load_floor_map(
        floor_map_path: str,
    ) -> np.ndarray:
        map_image_path = floor_map_path
        map_image = Image.open(map_image_path)

        # RGBもしくはL（グレースケール）モードに変換
        if map_image.mode == "RGBA":
            # アルファチャンネルを除去してRGBに変換
            map_image = map_image.convert("RGB")

        map_image = map_image.convert("L")

        return np.array(map_image, dtype=bool)

    # その点が歩行可能かどうかを判断する関数
    def is_passable(
        self,
        x: float,
        y: float,
    ) -> bool:
        epsilon = 1e-9  # 非常に小さい値
        # 不動小数点の切り捨てによる誤差を防ぐために、微小な値を足している
        # 例えば32.51の場合微小な値を足さないと3250.9999999999995となり、3250に切り捨てられてしまう
        row = int((x + epsilon) / self.dx)
        col = int((y + epsilon) / self.dy)

        #  numpy配列の範囲外にアクセスしようとした場合はFalseを返す
        if (
            row < 0
            or col < 0
            or row >= self.floor_map_data.shape[0]
            or col >= self.floor_map_data.shape[1]
        ):
            return False

        passable: bool = self.floor_map_data[row, col]

        return passable
