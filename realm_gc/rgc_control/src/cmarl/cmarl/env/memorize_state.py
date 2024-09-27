import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft
import matplotlib.pyplot as plt

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_mpe
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class MemorizeState(MultiAgentEnv):
    """
    The desired action of the agents is the observation of the first state.
    """

    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: State
        goal: State

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 5,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 8,
        "default_area_size": 1,
        "dist2goal": 0.01,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 256,
            max_travel: Optional[float] = None,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = MemorizeState.PARAMS["default_area_size"] if area_size is None else area_size
        super(MemorizeState, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

    @property
    def state_dim(self) -> int:
        return 1  # x

    @property
    def node_dim(self) -> int:
        return 3  # state dim (1) + indicator: agent: 01, goal: 10

    @property
    def edge_dim(self) -> int:
        return 1  # x_rel

    @property
    def action_dim(self) -> int:
        return 1  # vx

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 1, self.num_agents, 4 * self.params["car_radius"], None, self.max_travel)

        # states = jnp.array([[0.2], [0.4]])
        # goals = jnp.array([[1.5], [0.8]])

        env_states = self.EnvState(states, goals)

        return self.get_init_graph(env_states)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = action
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_agent_states = self.clip_state(next_agent_states)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward
        # each goal finds the nearest agent
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= ((action - (goals - agent_states))**2).mean()

        # agent_pos = agent_states
        # goal_pos = goals
        # dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1).min(axis=1)
        # reward -= (dist2goal ** 2).mean()
        # # goal reaching bonus
        # reward += jnp.where(dist2goal < self._params["dist2goal"], 1.0, 0.0).sum() * 0.01
        # # action penalty
        # reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.001

        # compute cost
        cost = jnp.zeros(()).astype(jnp.float32)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goals)

        info = {}

        return self.get_graph(next_state), reward, cost, done, info

    # def get_cost(self, graph: EnvGraphsTuple) -> Cost:
    #     agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
    #
    #     # collision between agents
    #     agent_pos = agent_states
    #     dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
    #     dist += jnp.eye(self.num_agents) * 1e6
    #     collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
    #     cost = collision.mean()
    #
    #     return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_mpe(rollout=rollout, video_path=video_path, side_length=self.area_size, dim=1, n_agent=self.num_agents,
                   n_rays=0, r=self.params["car_radius"], Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts, dpi=dpi,
                   **kwargs)

    def edge_blocks(self, state: EnvState) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.zeros((self.num_agents, self.num_agents))
        # agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        return [agent_agent_edges, agent_goal_edges]

    def init_edge_blocks(self, state: EnvState) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        return [agent_agent_edges, agent_goal_edges]

    def get_init_graph(self, state: EnvState) -> GraphsTuple:
        # node features
        n_nodes = self.num_agents * 2
        node_feats = jnp.zeros((self.num_agents * 2, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :1].set(state.agent)  # agent state
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, :1].set(state.goal)  # goal state
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats

        # node type
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(MemorizeState.GOAL)

        # edge blocks
        edge_blocks = self.init_edge_blocks(state)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal], axis=0),
        ).to_padded()

    def get_graph(self, state: EnvState) -> GraphsTuple:
        # node features
        n_nodes = self.num_agents * 2
        node_feats = jnp.zeros((self.num_agents * 2, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :1].set(state.agent)  # agent state
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, :1].set(state.goal)  # goal state
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats

        # node type
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(MemorizeState.GOAL)

        # edge blocks
        edge_blocks = self.edge_blocks(state)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.zeros(1)
        upper_lim = jnp.ones(1) * self.area_size
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(1) * -1.0
        upper_lim = jnp.ones(1)
        return lower_lim, upper_lim

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        return unsafe_agent
