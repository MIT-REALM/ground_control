import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional

from .lidar_env import LidarEnv
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .single_integrator_cover import SingleIntegratorCover
from .obstacle import Obstacle, Rectangle
from .plot import render_mpe
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class SingleIntNav(LidarEnv):
    """
    Agents are supposed to reach their own goals,
    while avoiding colliding with each other and obstacles.
    The agents are single integrators.
    """

    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.05, 0.15],
        "n_obs": 4,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8
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
        super(SingleIntNav, self).__init__(
            num_agents=num_agents,
            area_size=SingleIntNav.PARAMS["default_area_size"] if area_size is None else area_size,
            max_step=max_step,
            max_travel=max_travel,
            dt=dt,
            params=params
        )

    @property
    def reward_min(self) -> float:
        return -((self.area_size * np.sqrt(2)) * 0.01 - 0.001 - 0.0001) * self.max_episode_steps * 2.0

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        reward = jnp.zeros(()).astype(jnp.float32)

        # goal distance penalty
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        reward -= (dist2goal.mean()) * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward

    def edge_blocks(self, state: EnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        n_hits = self._params["top_k_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_pos_i = state.agent[i_agent]
            goal_pos_i = state.goal[i_agent]
            agent_goal_feats_i = agent_pos_i - goal_pos_i
            feats_norm = jnp.linalg.norm(agent_goal_feats_i, keepdims=True)
            safe_feats_norm = jnp.maximum(feats_norm, self.params["comm_radius"])
            coef = jnp.where(feats_norm > self.params["comm_radius"], self.params["comm_radius"] / safe_feats_norm, 1.0)
            agent_goal_feats_i = agent_goal_feats_i * coef
            agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))
        # id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        # agent_goal_mask = jnp.eye(self.num_agents)
        # agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        # feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        # comm_radius = self._params["comm_radius"]
        # safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        # coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        # agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        # agent_goal_edges = EdgeBlock(
        #     agent_goal_feats, agent_goal_mask, id_agent, id_goal
        # )

        # agent - obs connection
        agent_obs_edges = []
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges
