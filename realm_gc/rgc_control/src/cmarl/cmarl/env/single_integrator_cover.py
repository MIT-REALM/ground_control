import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_mpe
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class SingleIntegratorCover(MultiAgentEnv):
    """
    Agents are supposed to cover all the goals in the environment,
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
        "obs_len_range": [0.1, 0.3],
        "n_obs": 2,
        "default_area_size": 1.0,
        "dist2goal": 0.01,
        "top_k_rays": 8,
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
        area_size = SingleIntegratorCover.PARAMS["default_area_size"] if area_size is None else area_size
        super(SingleIntegratorCover, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)

    @property
    def state_dim(self) -> int:
        return 2  # x, y

    @property
    def node_dim(self) -> int:
        return 5  # state dim (2) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 2  # x_rel, y_rel

    @property
    def action_dim(self) -> int:
        return 2  # vx, vy

    @property
    def reward_min(self) -> float:
        return -30.0

    @property
    def reward_max(self) -> float:
        return 1.0

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
        # jax.debug.print('obstacles: {}', obstacles.n)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2 * self.params["car_radius"], obstacles, self.max_travel)  # change 4r to r
        env_states = self.EnvState(states, goals, obstacles)

        # get lidar data
        lidar_data = None
        if obstacles.n != 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states)  # (n_agent, top_k_rays, 2)

        return self.get_graph(env_states, lidar_data)

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
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_agent_states = self.clip_state(next_agent_states)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # get lidar data
        lidar_data = None
        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=obstacles,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
                max_returns=self._params["top_k_rays"],
            )
        )
        if obstacles.n != 0:
            lidar_data = get_lidar_vmap(agent_states)  # (n_agent, top_k_rays, 2)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph, lidar_data)

        assert reward.shape == tuple()
        # assert cost.shape == (self.n_cost,)
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goals, obstacles)
        if obstacles.n != 0:
            lidar_data_next = get_lidar_vmap(next_agent_states)
        else:
            lidar_data_next = None

        info = {}

        return self.get_graph(next_state, lidar_data_next), reward, cost, done, info

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)

        # each goal finds the nearest agent
        reward = jnp.zeros(()).astype(jnp.float32)
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1).min(axis=1)
        reward -= (dist2goal ** 2).mean()

        # # goal reaching bonus
        # reward += jnp.where(dist2goal < self._params["dist2goal"], 1.0, 0.0).sum() * 0.01

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.001

        return reward

    def get_cost(self, graph: GraphsTuple, lidar_data: Pos2d) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        # obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist
        # cost = jnp.maximum(cost_val, 0.0).mean()

        eps = 0.5
        agent_cost = jnp.where(agent_cost <= 0.0, agent_cost - eps, agent_cost + eps)
        agent_cost = jnp.clip(agent_cost, a_min=-1.0)  # (n_agent,)
        # agent_cost = agent_cost.max()
        # cost = jnp.where(cost >= 0.0, cost + eps, cost)

        # cost = cost.mean()

        # collision between agents and obstacles
        if graph.env_states.obstacle.n == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            # n_hits = self._params["top_k_rays"] * self.num_agents
            # is_lidar = jnp.logical_and(graph.senders >= self.num_agents * 2,
            #                            graph.senders < self.num_agents * 2 + n_hits)[:, None]
            # lidar_data = jnp.where(is_lidar, graph.edges, jnp.ones_like(graph.edges) * (self.params["comm_radius"] + 1))
            # obs_cost: Array = self.params["car_radius"] - jnp.linalg.norm(lidar_data[:, :2], axis=1)
            # jax.debug.breakpoint()
            # get_lidar_vmap = jax_vmap(
            #     ft.partial(
            #         get_lidar,
            #         obstacles=graph.env_states.obstacle,
            #         num_beams=self._params["n_rays"],
            #         sense_range=self._params["comm_radius"],
            #         max_returns=self._params["top_k_rays"],
            #     )
            # )
            # lidar_data = get_lidar_vmap(agent_pos)  # (n_agent, top_k_rays, 2)
            dist = jnp.linalg.norm(lidar_data - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)
            # obs_cost: Array = self.params["car_radius"] - dist  # (n_agent, top_k_rays)
        obs_cost = jnp.where(obs_cost <= 0.0, obs_cost - eps, obs_cost + eps)
        obs_cost = jnp.clip(obs_cost, a_min=-1.0)
        # obs_cost = obs_cost.max()

        # collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        # cost += collision.mean()
        # cost = jnp.array([agent_cost, obs_cost])
        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        # cost = jnp.concatenate([agent_cost[:, None], obs_cost], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_mpe(rollout=rollout, video_path=video_path, side_length=self.area_size, dim=2, n_agent=self.num_agents,
                   n_rays=self.params["top_k_rays"] if self.params["n_obs"] > 0 else 0,
                   r=self.params["car_radius"], cost_components=self.cost_components,
                   Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts, dpi=dpi, **kwargs)

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
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.ones((self.num_agents, self.num_agents))
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

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

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def get_graph(self, state: EnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        """
        lidar_data: (n_agent, top_k_rays, 2)
        """
        # node features
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :2].set(state.agent)  # agent state
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, :2].set(state.goal)  # goal state

        node_feats = node_feats.at[: self.num_agents, 4].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 3].set(1)  # goal feats
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[-n_hits:, 2].set(1)  # obs feats

        # node type
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(SingleIntegratorCover.GOAL)
        if self.params["n_obs"] > 0:
            node_type = node_type.at[-n_hits:].set(SingleIntegratorCover.OBS)

        # edge blocks
        if self.params["n_obs"] == 0:
            edge_blocks = self.edge_blocks(state)
            # lidar_data = None
        else:
            # get_lidar_vmap = jax_vmap(
            #     ft.partial(
            #         get_lidar,
            #         obstacles=state.obstacle,
            #         num_beams=self._params["n_rays"],
            #         sense_range=self._params["comm_radius"],
            #         max_returns=self._params["top_k_rays"],
            #     )
            # )
            # lidar_data = merge01(get_lidar_vmap(state.agent))
            lidar_data = merge01(lidar_data)
            # node_feats = node_feats.at[-n_hits:, :2].set(lidar_data)
            edge_blocks = self.edge_blocks(state, lidar_data)
            node_feats = node_feats.at[-n_hits:, :2].set(lidar_data)  # obs state

        # create graph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            states = jnp.concatenate([states, lidar_data], axis=0)
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.zeros(2)
        upper_lim = jnp.ones(2) * self.area_size
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
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

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        unsafe_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return unsafe_mask
