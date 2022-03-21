from __future__ import annotations

from dm_env import Environment, StepType, TimeStep, specs
import jax
from jax import numpy as jnp
from jax.numpy import ndarray as Array

from .grid import AgentEast, Goal, Grid, Point


class OnlyForwardMinigrid(Environment):
    def __init__(
        self,
        size: int,
        seed: int = 0,
        agent_pos: Point = None,
        goal_pos: Point = None,
    ):
        self.size: int = size
        self.grid: Grid = Grid.from_size(size, seed)
        self.init_agent_pos: Point = agent_pos
        self.goal_pos: Point = goal_pos
        self.reset()

    def observation_spec(self) -> specs.Array:
        return specs.BoundedArray(
            self.observation().shape, float, 0, 255, "observation"
        )

    def action_spec(self) -> specs.Array:
        return specs.DiscreteArray(len(self.actions()), int, "actions")

    def reward_spec(self) -> specs.Array:
        return specs.Array((1,), float, "rewards")

    def actions(self):
        return {
            0: (8, 2),  # rotate east and advance
            1: (9, 2),  # rotate south and advance
        }

    def observation(self) -> Array:
        return self.grid.as_rgb()

    def reward(self, obs: Array) -> float:
        if obs is None:
            obs = self.observation()
        goal_positions = self.grid.find_obj(Goal)
        agent_pos = self.grid.find_agent()
        is_in = jax.vmap(jnp.array_equal, in_axes=(None, 0))  # type: ignore
        r = jnp.any(is_in(agent_pos, goal_positions))
        return r

    def step_type(self, obs: Array) -> StepType:
        if obs is None:
            obs = self.observation()
        goal_positions = self.grid.find_obj(Goal)
        agent_pos = self.grid.find_agent()
        is_in = jax.vmap(jnp.array_equal, in_axes=(None, 0))  # type: ignore
        terminal = jnp.any(is_in(agent_pos, goal_positions))
        return StepType(terminal + 1)  # 0 start, 1 mid, 2 end

    def reset(self) -> TimeStep:
        # generate new grid
        self.grid = Grid.from_size(self.size)

        # place agent top left, exclude walls
        self.grid.place_obj(AgentEast, (1, 1))

        # place goal bottom right, exclude walls
        bottom, right = self.grid.tiles.shape
        self.grid.place_obj(Goal, (bottom - 2, right - 2))

        # pack timestep
        obs = self.observation()
        reward = self.reward(obs)
        discount = 1.0
        return TimeStep(StepType.FIRST, discount, reward, obs)

    def step(self, action: Array) -> TimeStep:
        actions = self.actions()[int(action)]
        self.grid.transition(actions)

        # pack timestep
        obs = self.observation()
        reward = self.reward(obs)
        step_type = StepType(int(reward == 1.0) + 1)
        discount = 1.0
        return TimeStep(step_type, discount, reward, obs)

    def render(self, mode: str = "rgb") -> Array:
        """
        Args:
            mode str: Choose an option between `rgb`, `categorical` and `ascii`

        Returns:
            A jax.numpy.ndarray with the required representations, of sizes
            (3, size, size) if rgb is requested, (size, size) otherwise.
        """
        mode = mode.lower()
        if mode == "rgb":
            return self.grid.as_rgb()
        elif mode == "categorical":
            return self.grid.as_categorical()

    def close(self) -> None:
        pass
