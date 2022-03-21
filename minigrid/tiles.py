
import logging
from typing import Dict, Type

import jax.numpy as jnp
from jax.numpy import ndarray as Array

from . import rendering


class Tile:
    ID: int = 0
    is_transparent: bool = True
    is_walkable: bool = True
    is_pickable: int = False
    as_rgb: Array = NotImplemented
    as_string: str = NotImplemented


_registry: Dict[int, Type[Tile]] = {}


def register_tile(idx: int):
    def register_tile_at_idx(cls: Type[Tile]):
        if idx in _registry:
            logging.warning(
                f"Tile with idx {idx} already register. Please use another idx."
            )
            return cls
        _registry[idx] = cls
        cls.ID = idx
        return cls

    return register_tile_at_idx


@register_tile(1)
class Empty(Tile):
    as_rgb: Array = rendering.block(rendering.WHITE)
    as_string: str = "/"


@register_tile(2)
class Wall(Tile):
    is_transparent: bool = False
    is_walkable: bool = False
    as_rgb: Array = rendering.block(rendering.GREY_100, 1.0)
    as_string: str = "|"


@register_tile(3)
class Floor(Tile):
    as_rgb: Array = rendering.floor()
    as_string: str = " "


@register_tile(4)
class DoorOpen(Tile):
    is_transparent: bool = False
    as_rgb: Array = NotImplemented
    as_string: str = "_"


@register_tile(14)
class DoorClosed(Tile):
    is_transparent: bool = False
    is_walkable: bool = False
    as_rgb: Array = NotImplemented
    as_string: str = "/"


@register_tile(5)
class Key(Tile):
    is_pickable = True
    as_rgb: Array = rendering.key(rendering.YELLOW)
    as_string: str = "K"


@register_tile(6)
class Ball(Tile):
    is_pickable = True
    as_rgb: Array = rendering.circle((0.5, 0.5), 0.31, rendering.BLUE)
    as_string: str = "o"


class Box(Tile):
    # TODO(ep)
    ...


@register_tile(8)
class Goal(Tile):
    as_rgb: Array = rendering.block(rendering.GREEN, 0.8)
    as_string: str = "*"


register_tile(9)
class Lava(Tile):
    as_rgb: Array = NotImplemented
    as_string: str = "§"


@register_tile(10)
class AgentEast(Tile):
    as_rgb: Array = jnp.rot90(rendering.agent(rendering.RED), k=1, axes=(0, 1))
    as_string: str = ">"


@register_tile(11)
class AgentSouth(Tile):
    as_rgb: Array = jnp.rot90(rendering.agent(rendering.RED), k=0, axes=(0, 1))
    as_string: str = "V"


@register_tile(12)
class AgentWest(Tile):
    as_rgb: Array = jnp.rot90(rendering.agent(rendering.RED), k=-1, axes=(0, 1))
    as_string: str = "<"


@register_tile(13)
class AgentNorth(Tile):
    as_rgb: Array = jnp.rot90(rendering.agent(rendering.RED), k=-2, axes=(0, 1))
    as_string: str = "Λ"
