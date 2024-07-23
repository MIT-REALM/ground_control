import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class SingleIntegratorAdaptEdge(MultiAgentEnv):

    AGENT = 0
    GOAL = 1
    OBS = 2
    SLOW_MOV_OBS = 3
    FAST_MOV_OBS = 4

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle
        mov_obs: State

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 8,
        "n_mov_obs": 4,
        "slow_mov_obs_speed": 0.1,
        "fast_mov_obs_speed": 0.2,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            node_feat: int = 4,
            params: dict = None
    ):
        super(SingleIntegratorAdaptEdge, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self._A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32) * self._dt + np.eye(self.state_dim)
        self._B = np.array([[1.0, 0.0], [0.0, 1.0]]) * self._dt
        self._Q = np.eye(self.state_dim) * 2
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        self.create_obstacles = jax_vmap(Rectangle.create)
        # self.slow_mov_obs_vel = None
        # self.fast_mov_obs_vel = None
        self.mov_obs_vel = None
        self._node_feat = node_feat
        self.safe_slow_obs_dist = 2 * self._params["car_radius"]
        self.safe_fast_obs_dist = 4 * self._params["car_radius"]

    @property
    def state_dim(self) -> int:
        return 2  # x, y

    @property
    def node_dim(self) -> int:
        # return 6  
        return self._node_feat
        # first five features indicator: agent: 00001, goal: 00010, stationary obstacle: 00100, slow moving obstacle: 01000, fast moving obstacle: 10000
        # last two features: v_obs and r_obs

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # vx, vy

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        n_rng_mov_obs = self._params["n_mov_obs"]
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

        mov_obs_key, key = jr.split(key, 2)
        mov_obs_pos = jr.uniform(mov_obs_key, (n_rng_mov_obs, 2), minval=0, maxval=self.area_size)
        
        mov_obs_vel_key, key = jr.split(key, 2)
        slow_mov_obs_vel = jr.uniform(mov_obs_vel_key, (n_rng_mov_obs // 2, 2), minval=-self._params["slow_mov_obs_speed"], maxval=self._params["slow_mov_obs_speed"])
        # self.slow_mov_obs_vel = slow_mov_obs_vel
        
        mov_obs_vel_key, key = jr.split(key, 2)
        fast_mov_obs_vel = jr.uniform(mov_obs_vel_key, (n_rng_mov_obs // 2, 2), minval=-self._params["fast_mov_obs_speed"], maxval=self._params["fast_mov_obs_speed"])
        # self.fast_mov_obs_vel = fast_mov_obs_vel
        
        self.mov_obs_vel = jnp.concatenate([slow_mov_obs_vel, fast_mov_obs_vel], axis=0)
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel, mov_obs_pos)

        env_states = self.EnvState(states, goals, obstacles, mov_obs_pos)

        return self.get_graph(env_states)

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
        mov_obstacles = graph.env_states.mov_obs
        next_mov_obstacles = mov_obstacles + self.mov_obs_vel * self.dt
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goals, obstacles, next_mov_obstacles)

        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            agent_pos = agent_states
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        mov_obstacles = graph.env_states.mov_obs

        # collision between agents
        agent_pos = agent_states
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()
        
        # collision between agents and slow moving obstacles
        collision_mov_obs = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(mov_obstacles[:self._params["n_mov_obs"] // 2], 0), axis=-1)
        collision = (self.safe_slow_obs_dist > collision_mov_obs).any(axis=1)
        cost += collision.mean()

        # collision between agents and fast moving obstacles
        collision_mov_obs = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(mov_obstacles[self._params["n_mov_obs"] // 2:], 0), axis=-1)
        collision = (self.safe_fast_obs_dist > collision_mov_obs).any(axis=1)
        cost += collision.mean()
        
        return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            n_mov_obs=self.params["n_mov_obs"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: Pos2d) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_vel = self.mov_agent_vel_pred(state=agent_pos)
        rel_vel = agent_vel[:, None, :] - agent_vel[None, :, :]
        pos_diff = jnp.concatenate([pos_diff, rel_vel], axis=-1)
        # agent_speed_feat = jnp.linalg.norm(self.action_lim()[1]) * jnp.eye(self.num_agents)
        # breakpoint()
        # jax.debug.breakpoint()
        # agent_safe_dist = jnp.ones((self.num_agents, self.num_agents)) * self._params["car_radius"] * 2
        # pos_feat = jnp.concatenate([pos_diff, agent_speed_feat[..., None], agent_safe_dist[..., None]], axis=-1)
        # agent_agent_edges = EdgeBlock(pos_feat, agent_agent_mask, id_agent, id_agent)
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
        agent_goal_rel_vel = jnp.zeros((self.num_agents, self.num_agents, 2))
        agent_goal_feats = jnp.concatenate([agent_goal_feats, agent_goal_rel_vel], axis=-1)
        # agent_goal_feats = jnp.concatenate([agent_goal_feats, jnp.zeros((self.num_agents, self.num_agents, 2))], axis=-1)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            obs_rel_vel = jnp.zeros((self._params["n_rays"], 2)) - agent_vel[i]
            lidar_feats = jnp.concatenate([lidar_feats, obs_rel_vel], axis=-1)
            # lidar_feats = jnp.concatenate([lidar_feats, jnp.zeros((self._params["n_rays"], 1)), jnp.ones((self._params["n_rays"], 1)) * self._params["car_radius"] * 1], axis=-1)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )
        
        # agent - mov obs connection
        id_mov_obs = jnp.arange(self.num_agents * 2 + n_hits, self.num_agents * 2 + n_hits + self._params["n_mov_obs"])
        agent_mov_obs_edges = []
        mov_obs_pos = state.mov_obs
        mov_obs_vel = self.mov_obs_vel_pred(state=state.mov_obs)
        for i in range(self.num_agents):
            # id_mov_obs = jnp.arange(self.num_agents * 2 + n_hits, self.num_agents * 2 + n_hits + self._params["n_mov_obs"])
            agent_mov_obs_pos = agent_pos[i, :] - mov_obs_pos
            agent_mov_obs_dist = jnp.linalg.norm(agent_mov_obs_pos, axis=-1)
            agent_mov_obs_mask = jnp.ones((1, self._params["n_mov_obs"]))
            mov_obs_rel_vel = mov_obs_vel - agent_vel[i]
            agent_mov_obs_pos = jnp.concatenate([agent_mov_obs_pos, mov_obs_rel_vel], axis=-1)
            # agent_mov_obs_pos = jnp.concatenate([agent_mov_obs_pos, jnp.ones((self._params["n_mov_obs"], 1)) * self._params["mov_obs_speed"], jnp.ones((self._params["n_mov_obs"], 1)) * self._params["car_radius"] * 4], axis=-1)
            agent_mov_obs_mask = jnp.logical_and(jnp.less(agent_mov_obs_dist, self._params["comm_radius"]), agent_mov_obs_mask)
            agent_mov_obs_edges.append(
                EdgeBlock(agent_mov_obs_pos[None, :, :], agent_mov_obs_mask, id_agent[i][None], id_mov_obs)
            )
        # mov_obs_pos = state.mov_obs
        # agent_mov_obs_pos = agent_pos[:, None, :] - mov_obs_pos[None, :, :]
        # agent_mov_obs_dist = jnp.linalg.norm(agent_mov_obs_pos, axis=-1)
        # agent_mov_obs_mask = jnp.less(agent_mov_obs_dist, self._params["comm_radius"])
        # agent_mov_obs_edges = EdgeBlock(agent_mov_obs_pos, agent_mov_obs_mask, id_agent, id_mov_obs)
        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges + agent_mov_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.zeros_like(state)
        g = jnp.eye(state.shape[1])
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2
        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        # edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)
        graph_edges = graph.edges
        graph_edges = graph_edges.at[:, :2].set(edge_feats[:, :2] * coef)
        return graph._replace(edges=graph_edges, states=state)

    def get_graph(self, state: EnvState) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        num_mov_obs = self._params["n_mov_obs"]
        n_nodes = 2 * self.num_agents + n_hits + num_mov_obs
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits + num_mov_obs, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, 4].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 3].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits-num_mov_obs:-num_mov_obs, 2].set(1)  # obs feats
        node_feats = node_feats.at[-num_mov_obs:-num_mov_obs//2, 1].set(1)  # slow mov obs feats
        node_feats = node_feats.at[-num_mov_obs//2:, 0].set(1)  # fast mov obs feats
        
        if self.node_dim > 5:
            agent_speed = jnp.linalg.norm(self.action_lim()[1])
            agent_safe_dist = self._params["car_radius"] * 2
            obs_speed = 0.0
            obs_safe_dist = self._params["car_radius"]
            node_feats = node_feats.at[:self.num_agents, 5].set(agent_speed)  # v_obs feats
            node_feats = node_feats.at[:self.num_agents, 6].set(agent_safe_dist)  # r_obs feats
            node_feats = node_feats.at[-n_hits-num_mov_obs:-num_mov_obs, 5].set(obs_speed)  # v_obs feats
            node_feats = node_feats.at[-n_hits-num_mov_obs:-num_mov_obs, 6].set(obs_safe_dist)  # r_obs feats
            slow_move_obs_speed = self._params["slow_mov_obs_speed"]
            node_feats = node_feats.at[-num_mov_obs:-num_mov_obs//2, 5].set(slow_move_obs_speed)  # v_obs feats
            node_feats = node_feats.at[-num_mov_obs:-num_mov_obs//2, 6].set(self.safe_slow_obs_dist)  # r_obs feats
            fast_move_obs_speed = self._params["fast_mov_obs_speed"]
            node_feats = node_feats.at[-num_mov_obs//2:, 5].set(fast_move_obs_speed)  # v_obs feats
            node_feats = node_feats.at[-num_mov_obs//2:, 6].set(self.safe_fast_obs_dist)  # r_obs feats

        # node type
        # node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(SingleIntegratorAdaptEdge.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(SingleIntegratorAdaptEdge.GOAL)
        node_type = node_type.at[-n_hits-self._params["n_mov_obs"]: -self.params["n_mov_obs"]].set(SingleIntegratorAdaptEdge.OBS)
        node_type = node_type.at[-self._params["n_mov_obs"]:-self._params["n_mov_obs"] // 2].set(SingleIntegratorAdaptEdge.SLOW_MOV_OBS)
        node_type = node_type.at[-self._params["n_mov_obs"] // 2:].set(SingleIntegratorAdaptEdge.FAST_MOV_OBS)
        
        # jax.debug.breakpoint()
        # breakpoint()

        # edge blocks
        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent))
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data, state.mov_obs], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.ones(2) * -jnp.inf
        upper_lim = jnp.ones(2) * jnp.inf
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)

        return next_graph

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 2.5)
        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)
        )
        
        mov_obs_pos = graph.env_states.mov_obs
        
        slow_mov_obs_dist = agent_pos[:, None, :] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        slow_mov_obs_mask = jnp.greater(slow_mov_obs_dist, self.safe_slow_obs_dist + self._params["car_radius"] * 1)

        fast_mov_obs_dist = agent_pos[:, None, :] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        fast_mov_obs_mask = jnp.greater(fast_mov_obs_dist, self.safe_fast_obs_dist + self._params["car_radius"] * 1.5)
        
        safe_mov_obs = jnp.logical_and(slow_mov_obs_mask, fast_mov_obs_mask)

        safe_mask = jnp.logical_and(safe_agent, safe_obs)
        
        safe_mask = jnp.logical_and(safe_mask, safe_mov_obs)

        return safe_mask

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

        # agents are colliding with slow moving obstacles
        mov_obs_pos = graph.env_states.mov_obs
        slow_mov_obs_dist = agent_pos[:, None, :] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        unsafe_slow_mov_obs = jnp.less(slow_mov_obs_dist, self.safe_slow_obs_dist).any(axis=1)
        
        # agents are colliding with fast moving obstacles
        fast_mov_obs_dist = agent_pos[:, None, :] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        unsafe_fast_mov_obs = jnp.less(fast_mov_obs_dist, self.safe_fast_obs_dist).any(axis=1)
        
        unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_slow_mov_obs)
        unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_fast_mov_obs)
        # agents are colliding with moving obstacles
        # mov_obs_pos = graph.env_states.mov_obs
        # mov_obs_dist = agent_pos[:, None, :] - mov_obs_pos[None, :, :]
        # mov_obs_dist = jnp.linalg.norm(mov_obs_dist, axis=-1)
        # unsafe_mov_obs = jnp.less(mov_obs_dist, self._params["car_radius"] * 4).any(axis=1)
        # unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_mov_obs)
        
        return unsafe_mask

    def collision_mask(self, graph: GraphsTuple) -> Array:
        return self.unsafe_mask(graph)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach
