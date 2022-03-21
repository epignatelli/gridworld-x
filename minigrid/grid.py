"""This module is heavily inspired by
https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py
This refactor adds GPU support
"""
from __future__ import annotations

import logging
from functools import wraps
from re import A
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.numpy import ndarray as Array
from jax.random import KeyArray
from jax.tree_util import register_pytree_node_class

from . import rendering, tiles
from .tiles import AgentEast, AgentNorth, Empty, Floor, Goal, Tile

Point = Union[Tuple[float, float], Array, None]


ActionFunction = Callable[[Any], None]
ACTIONS: Dict[int, Callable[[Any], None]] = {}


def register_action(idx: int) -> Callable[[ActionFunction], ActionFunction]:
    def register_action_ad_idx(action_fn: ActionFunction):
        @wraps(action_fn)
        def fn(grid: Any) -> None:
            return action_fn(grid)

        if idx in ACTIONS:
            logging.warning(
                f"Action with idx {idx} already register. Please use another idx."
            )
            return fn
        ACTIONS[idx] = action_fn
        return fn

    return register_action_ad_idx


DIRECTIONS: Dict[int, Array] = {
    # Pointing east
    0: jnp.array((1, 0), dtype=int),
    # Pointing south
    1: jnp.array((0, 1)),
    # Pointing west
    2: jnp.array((-1, 0)),
    # Pointing north
    3: jnp.array((0, -1)),
}


@register_pytree_node_class
class Grid:
    """Square grid"""

    _tiles_registry = tiles._registry

    def __init__(
        self,
        size: int,
        tiles: Array,
        seed: int,
        key: KeyArray,
    ):
        assert size > 0, f"Size must be a positive number, got {size} instead"
        assert size == len(
            tiles
        ), f"Got incompatible size with tiles {size}, expected {len(tiles)}"
        # static:
        self.width: int = size
        self.height: int = size
        self.seed: int = seed
        # dynamic:
        self.tiles: Array = tiles
        self.key: KeyArray = key

    def tree_flatten(self):
        return ((self.tiles, self.key), (self.seed,))

    @classmethod
    def tree_unflatten(cls, aux: Tuple[int], children: Tuple[Array, KeyArray]):
        tiles, key = children
        seed = aux[0]
        return cls(len(tiles), tiles, seed, key)

    @classmethod
    def from_size(cls, size: int, seed: int = 0) -> Grid:
        # `3` stands for floor
        tiles: Array = jnp.ones((size, size), dtype=int) * 3
        key: KeyArray = jax.random.PRNGKey(seed)
        return cls(size, tiles, seed, key)

    def transition(self, actions: List[int]):
        for a in actions:
            act = ACTIONS[int(a)]
            act(self)
        return

    def next_key(self, n_keys: int = 1) -> KeyArray:
        self.key, k = jax.random.split(self.key, n_keys + 1)
        return k

    def get(self, pos: Point) -> int:
        if pos is None:
            raise ValueError("Invalid position to inspect {}".format(pos))
        obj = self.tiles[tuple(pos)]
        return obj

    def set(self, pos: Point, id: int):
        if pos is None:
            raise ValueError("Invalid position to inspect {}".format(pos))
        pos = tuple(pos)
        self.tiles = self.tiles.at[pos].set(id)
        return

    def is_free(self, pos: Point) -> bool:
        if pos is None:
            return False
        return self.tiles[tuple(pos)] in (Empty.ID, Floor.ID)

    def rand_pos(self) -> Array:
        k = self.next_key()
        return jax.random.randint(k, (2,), 1, self.tiles.shape[0] - 2)

    def free_pos(self) -> Array:
        pos, i = self.rand_pos(), 0
        available_cells = ((self.tiles.shape[0] - 2) * 2) - 1
        while i < available_cells:
            i += 1
            pos = self.rand_pos()
            if not self.is_free(pos):
                continue
            return pos
        raise ValueError(f"There are no positions availble on the grid")

    def find_obj(self, obj: Type[Tile]) -> Sequence[Point]:
        """
        Args:
            obj Type[Tile]: The type of the tiles to search for
        Returns:
            A two-dimensional `Array` of shape (N, 2), containing
            the N obj
        """
        mask = self.tiles == obj.ID
        return jnp.asarray(jnp.nonzero(mask)).T

    def find_agent(self) -> Point:
        mask = jnp.logical_and(AgentEast.ID <= self.tiles, self.tiles <= AgentNorth.ID)
        return jnp.asarray(jnp.nonzero(mask, size=1, fill_value=0)).T[-1]

    def place_obj(self, obj: Type[Tile], pos: Point = None, force: bool = False):
        if 10 <= obj.ID <= 13:
            return self.place_agent(pos, force=force)
        # if no pos is specified, find a suitable one
        if pos is None:
            pos = self.free_pos()
        # if position is not free, and we don't force placing
        if not force and not self.is_free(pos):
            pos = self.free_pos()
        return self.set(pos, obj.ID)

    def place_agent(self, pos: Point = None, direction: int = 0, force: bool = False):
        if not (0 <= direction <= 3):
            raise ValueError(
                "agent's `direction` must be within `0` and `3`, got {} instaed".format(
                    direction
                )
            )
        # if no pos is specified, find a suitable one
        if pos is None or (not force and not self.is_free(pos)):
            pos = self.free_pos()

        # update agent's position
        self.set(pos, AgentEast.ID + direction)
        return

    def visibility_mask(self, radius: int) -> Array:
        a = self.find_agent()
        shape = (self.width, self.height)
        if a is None or radius == -1:
            return jnp.ones(shape)

        a_x, a_y = a
        directions = {
            0: (0, radius),
            1: (radius, 0),
            2: (a_x - 2 * radius, a_y - radius),
            3: (a_x - radius, a_y - 2 * radius)
        }
        direction = self.tiles[tuple(a)] - AgentEast.ID
        a_x, a_y = directions[direction]
        b_x, b_y = (a_x + 2 * radius, a_y + 2 * radius)
        x, y = jnp.mgrid[: self.width :, : self.height :]
        mask = jnp.logical_and(x >= a_x, x <= b_x)
        mask = jnp.logical_and(mask, y >= a_y)
        mask = jnp.logical_and(mask, y <= b_y)
        return mask

    def as_rgb(self, visibility_radius: int = -1):
        # TODO(ep): optimise this crap
        mask = self.visibility_mask(visibility_radius).reshape(self.width**2)
        tiles = self.tiles.reshape(self.width**2)
        tiles = [
            rendering.highlight(
                self._tiles_registry[int(tile_idx)].as_rgb, mask[i] * 0.3
            )
            for i, tile_idx in enumerate(tiles)
        ]

        images = jnp.concatenate(
            [
                jnp.concatenate(tiles[i * self.width : (i + 1) * self.width], axis=0)
                for i in range(self.width)
            ],
            axis=1,
        )
        return jnp.asarray(images, dtype=jnp.uint8)

    def as_categorical(self):
        return jnp.asarray(self.tiles, dtype=int)

    @register_action(0)
    def rotate_counterclockwise(self) -> None:
        agent_pos = self.find_agent()
        agent_id: int = self.tiles[agent_pos]
        rotated_agent_id = (agent_id - 1 - 10) % 4 + 10
        return self.set(agent_pos, rotated_agent_id)

    @register_action(1)
    def rotate_clockwise(self) -> None:
        agent_pos = self.find_agent()
        agent_id = self.tiles[agent_pos]
        rotated_agent_id = (agent_id + 1 - 10) % 4 + 10
        return self.set(agent_pos, rotated_agent_id)

    @register_action(2)
    def advance(self) -> None:
        current_pos = self.find_agent()
        agent_id = self.get(current_pos)
        direction = agent_id - AgentEast.ID
        dir_vector = DIRECTIONS[int(direction)]
        new_pos = current_pos + dir_vector
        tile_id = int(self.get(new_pos))
        if not self._tiles_registry[tile_id].is_walkable:
            return

        self.place_agent(new_pos, direction)
        self.place_obj(Floor, current_pos, True)
        return

    @register_action(3)
    def pickup(self) -> None:
        raise NotImplementedError

    @register_action(4)
    def drop(self) -> None:
        raise NotImplementedError

    @register_action(5)
    def toggle(self) -> None:
        raise NotImplementedError

    @register_action(6)
    def noop(self) -> None:
        return

    @register_action(7)
    def rotate_north(self) -> None:
        agent_pos = self.find_agent()
        # east direction is categorical `3`
        return self.place_agent(agent_pos, 3)

    @register_action(8)
    def rotate_east(self) -> None:
        agent_pos = self.find_agent()
        # east direction is categorical `0`
        return self.place_agent(agent_pos, 0)

    @register_action(9)
    def rotate_south(self) -> None:
        agent_pos = self.find_agent()
        # east direction is categorical `1`
        return self.place_agent(agent_pos, 1)

    @register_action(10)
    def rotate_west(self) -> None:
        agent_pos = self.find_agent()
        # east direction is categorical `2`
        return self.place_agent(agent_pos, 2)
