import pytest

from palkia.core.map.floor_map import FloorMap


@pytest.fixture
def floor_map() -> FloorMap:
    return FloorMap(
        floor_name="Test Floor",
        floor_map_path="../data/floor_maps/test_floor.png",
        dx=0.01,
        dy=0.01,
    )


def test_is_passable(floor_map: FloorMap) -> None:
    # 歩行可能な場所
    assert floor_map.is_passable(10.01, 15.02)
    assert floor_map.is_passable(0.0, 0.0)

    # 歩行不可能な場所
    assert not floor_map.is_passable(-0.01, 15.02)
    assert not floor_map.is_passable(10.01, 30.01)


def test_out_of_bounds(floor_map: FloorMap) -> None:
    # 範囲外の座標
    assert not floor_map.is_passable(-1, 0)
    assert not floor_map.is_passable(0, -1)
    assert not floor_map.is_passable(100, 100)
