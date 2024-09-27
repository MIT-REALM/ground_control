import jax.numpy as jnp
import numpy as np
import jax
import jax.random as jr

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from .mpe import MPE, MPEEnvState, MPEEnvGraphsTuple
from .obstacle import Obstacle, Rectangle
from .mpe_spread import MPESpread
from .mpe_nav import MPENav
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng
from ..utils.utils import merge01


class PointLidarNav(MPENav):

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_obs": 12,
        "obs_radius": 0.05,
        "obs_len_range": [0.05, 0.15],
        "default_area_size": 1.5,
        "dist2goal": 0.01
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            max_travel: Optional[float] = None,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = PointLidarNav.PARAMS["default_area_size"] if area_size is None else area_size
        super(PointLidarNav, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Rectangle.create)

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"] // 4
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"] // 4, 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
        points = merge01(obstacles.points)
        obs = jnp.concatenate([points, jnp.zeros_like(points)], axis=1)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 4 * self.params["car_radius"], obstacles, self.max_travel)
        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros_like(goals)], axis=1)
        # env_states = LidarEnvState(states, goals, obstacles)
        env_states = MPEEnvState(states, goals, obs)

        # get lidar data
        # lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states)

