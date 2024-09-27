import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractproperty, abstractmethod

from jaxtyping import Float

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_mpe
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle
    mov_obs: Optional[Obstacle] = None

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]


class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2
    SLOW_MOV_OBS = 3
    FAST_MOV_OBS = 4

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "n_mov_obs": 4,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
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
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 10  # state dim (5) + indicator: agent: 00001, goal: 00010, obstacle: 00100, slow mov obs: 01000. fast mov obs: 10000

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_vel, vy_vel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @abstractproperty
    def reward_min(self) -> float:
        pass

    @property
    def reward_max(self) -> float:
        return 0.5

    @property
    def cost_min(self) -> float:
        return -1.0

    @property
    def cost_max(self) -> float:
        return 1.0

    @property
    def n_cost(self) -> int:
        return 4

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", 'slow mov obs collisions', 'fast mov obs collisions'

    def reset(self, key: Array) -> GraphsTuple:
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

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles, self.max_travel)
        states = jnp.concatenate(
            [states, jnp.zeros((self.num_agents, self.state_dim - states.shape[1]), dtype=states.dtype)], axis=1)
        goals = jnp.concatenate(
            [goals, jnp.zeros((self.num_goals, self.state_dim - goals.shape[1]), dtype=goals.dtype)], axis=1)
        env_states = LidarEnvState(states, goals, obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        if self.params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 2)
        return lidar_data

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """By default, use double integrator dynamics"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False, iter=0
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.env_states.agent
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle if self.params['n_obs'] > 0 else None
        n_mov_obs = self._params["n_mov_obs"]
        
        if n_mov_obs > 0:
            mov_obs = graph.env_states.mov_obs
            mov_obs_pos = mov_obs[:, :2]
            mov_obs_vel = mov_obs[:, 2:4]
            
            mov_obs_max_x_mask = mov_obs_pos[:, 0] > self.area_size
            mov_obs_min_x_mask = mov_obs_pos[:, 0] < 0
            mov_obs_max_y_mask = mov_obs_pos[:, 1] > self.area_size
            mov_obs_min_y_mask = mov_obs_pos[:, 1] < 0
            
            mov_obs_vel = mov_obs_vel.at[:, 0].set(jnp.where(mov_obs_max_x_mask, -mov_obs_vel[:, 0], mov_obs_vel[:, 0]))
            mov_obs_vel = mov_obs_vel.at[:, 0].set(jnp.where(mov_obs_min_x_mask, -mov_obs_vel[:, 0], mov_obs_vel[:, 0]))
            mov_obs_vel = mov_obs_vel.at[:, 1].set(jnp.where(mov_obs_max_y_mask, -mov_obs_vel[:, 1], mov_obs_vel[:, 1]))
            mov_obs_vel = mov_obs_vel.at[:, 1].set(jnp.where(mov_obs_min_y_mask, -mov_obs_vel[:, 1], mov_obs_vel[:, 1]))
            
            mov_obs_pos += mov_obs_vel * self.dt
            new_mov_obs = jnp.concatenate([mov_obs_pos, mov_obs_vel, jnp.zeros((n_mov_obs, 1))], axis=1)
        else:
            new_mov_obs = None
        # calculate next states
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        
        # if iter > 0:
        # orig_goals = jnp.array([[-self.area_size, 0.], [0., -self.area_size]]) + 0.1
    
        # orig_goals = orig_goals / 2 + self.area_size / 2
    
        # new_goals1 = orig_goals[0, :] * (1 - iter) + jnp.array([self.area_size, self.area_size / 2]) * iter
        # new_goals2 = orig_goals[1, :] * (1 - iter) + jnp.array([self.area_size / 2, self.area_size]) * iter
        # new_goals = jnp.concatenate([new_goals1[None, :], new_goals2[None, :]], axis=0)
        # # jax.debug.breakpoint()
        # new_goals = jnp.concatenate([new_goals, jnp.zeros((2, 3), dtype=goals.dtype)], axis=1)
        # new_goals = jnp.concatenate([new_goals, jnp.zeros((self.num_goals - 2, 5), dtype=goals.dtype)], axis=0)
        
        # goals = jnp.where(iter > 0, new_goals, goals)
        
        next_state = LidarEnvState(next_agent_states, goals, obstacles, new_mov_obs)
        lidar_data_next = self.get_lidar_data(next_agent_states, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        # reward = self.get_reward(graph, action)
        # cost = self.get_cost(graph)
        # assert reward.shape == tuple()

        return self.get_graph(next_state, lidar_data_next), 0.0, 0.0, done, info

    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.env_states.agent

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        # collision between agents and moving obstacles
        if self.params["n_mov_obs"] > 0:
            slow_mov_obs_pos = graph.type_states(type_idx=3, n_type=self.params["n_mov_obs"] // 2)[:, :2]
            dist = jnp.linalg.norm(slow_mov_obs_pos - agent_pos[:, None, :], axis=-1)
            slow_mov_obs_cost = self.params["car_radius"] * 2 - dist.min(axis=1)
            fast_mov_obs_pos = graph.type_states(type_idx=4, n_type=self.params["n_mov_obs"] // 2)[:, :2]
            dist = jnp.linalg.norm(fast_mov_obs_pos - agent_pos[:, None, :], axis=-1)
            fast_mov_obs_cost = self.params["car_radius"] * 4 - dist.min(axis=1)
        else:
            slow_mov_obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
            fast_mov_obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
            
        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None], slow_mov_obs_cost[:, None], fast_mov_obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

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
                   Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts, n_goal=self.num_goals, dpi=dpi, **kwargs)

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass

    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        
        n_mov_obs = self.params["n_mov_obs"]
        
        n_nodes = self.num_agents + self.num_goals + n_hits + n_mov_obs

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + n_hits + n_mov_obs, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits - n_mov_obs:-n_mov_obs, :2].set(lidar_data)
        node_feats = node_feats.at[-n_mov_obs:, :4].set(state.mov_obs[:, :4])
        
        # indicators
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 4].set(1.)  # agent
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 3].set(1.))  # goal
        
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits-n_mov_obs: -n_mov_obs, self.state_dim + 2].set(1.)  # obs feats
        
        node_feats = node_feats.at[-n_mov_obs:-n_mov_obs//2, self.state_dim + 1].set(1.)  # slow mov obs
        node_feats = node_feats.at[-n_mov_obs//2:, self.state_dim].set(1.)  # fast mov obs

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(LidarEnv.GOAL)
        
        if n_hits > 0:
            node_type = node_type.at[-n_hits-n_mov_obs:-n_mov_obs].set(LidarEnv.OBS)
            node_type = node_type.at[-n_mov_obs:-n_mov_obs//2].set(LidarEnv.SLOW_MOV_OBS)
            node_type = node_type.at[-n_mov_obs//2:].set(LidarEnv.FAST_MOV_OBS)
        else:
            node_type = node_type.at[-n_mov_obs:-n_mov_obs//2].set(LidarEnv.SLOW_MOV_OBS)
            node_type = node_type.at[-n_mov_obs//2:].set(LidarEnv.FAST_MOV_OBS)
            
        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]))], axis=1)
            states = jnp.concatenate([states, lidar_states, state.mov_obs], axis=0)
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states
        ).to_padded()

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"]) if self.params["n_obs"] > 0 else None

        # calculate next graph
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goals, obstacles], axis=0)

        next_graph = graph._replace(
            states=next_states,
            nodes=graph.nodes.at[:self.num_agents, :self.state_dim].set(next_agent_states),
            edges=next_states[graph.receivers] - next_states[graph.senders]
        )
        return next_graph

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        cost = self.get_cost(graph)
        return jnp.any(cost >= 0.0, axis=-1)
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def goal_mask(self, graph: GraphsTuple) -> Array:
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)[:, :2]
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]    
        reach_rate = jnp.linalg.norm(goals - agent_pos, axis=-1) < self.params["dist2goal"]
        reach_rate = jnp.mean(reach_rate, axis=-1)
        return reach_rate
    
    def slow_mov_obs_collision_mask(self, graph: GraphsTuple) -> Array:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        mov_obs = graph.type_states(type_idx=3, n_type=self.params["n_mov_obs"] // 2)
        agent_pos = agent_states[:, :2]
        mov_obs_pos = mov_obs[:, :2]
        dist = jnp.linalg.norm(mov_obs_pos - agent_pos[:, None, :], axis=-1)
        return jnp.less(dist, self.params["car_radius"] * 2)

    def fast_mov_obs_collision_mask(self, graph: GraphsTuple) -> Array:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        mov_obs = graph.type_states(type_idx=4, n_type=self.params["n_mov_obs"] //2)
        agent_pos = agent_states[:, :2]
        mov_obs_pos = mov_obs[:, :2]
        dist = jnp.linalg.norm(mov_obs_pos - agent_pos[:, None, :], axis=-1)
        return jnp.less(dist, self.params["car_radius"] * 4)