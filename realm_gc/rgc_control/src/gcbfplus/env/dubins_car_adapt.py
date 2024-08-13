import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos2d, Reward, State
from ..utils.utils import merge01
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
# from .plot import render_video
from .utils import get_lidar, inside_obstacles, get_node_goal_rng

def render_video(**args):
    pass

class DubinsCarAdapt(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2
    SLOW_MOV_OBS = 3
    FAST_MOV_OBS = 4

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle
        mov_obs: State

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 16,
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
            mov_obs_speed: float = None,
            mov_obs_at_infty: bool = False,
            station_obs_at_infty: bool = False,
            params: dict = None,
            use_stop_mask: bool = False,
    ):
        super(DubinsCarAdapt, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Rectangle.create)
        self.enable_stop = True
        self.mov_obs_vel = None
        self.mov_obs_speed = mov_obs_speed
        self._params['slow_mov_obs_speed'] = self.mov_obs_speed if self.mov_obs_speed is not None else self._params['slow_mov_obs_speed']
        self._params['fast_mov_obs_speed'] = 2 * self.mov_obs_speed if self.mov_obs_speed is not None else self._params['fast_mov_obs_speed']
        self._node_feat = node_feat
        self.safe_slow_obs_dist = 2 * self._params["car_radius"]
        self.safe_fast_obs_dist = 3 * self._params["car_radius"]
        self.mov_obs_at_infty = mov_obs_at_infty
        self.station_obs_at_infty = station_obs_at_infty
        self.use_stop_mask = use_stop_mask
        
    @property
    def state_dim(self) -> int:
        return 4  # x, y, theta, v

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
        return 2  # omega, acc
    
    def mov_obsvel(self) -> Array:
        return self.mov_obs_vel
    
    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        n_rng_mov_obs = self._params["n_mov_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (self._params["n_obs"], 2), minval=0, maxval=self.area_size)
        
        if self.station_obs_at_infty:
            obs_pos = obs_pos.at[:, :2].set(jnp.array([self.area_size, self.area_size]) * 100)
            
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (self._params["n_obs"],), minval=0, maxval=2 * np.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        mov_obs_key, key = jr.split(key, 2)
        mov_obs_pos = jr.uniform(mov_obs_key, (n_rng_mov_obs, 4), minval=0, maxval=self.area_size)
        
        # mov_obs_pos = mov_obs_pos.at[:, 2:].set(0)
        if self.mov_obs_at_infty:
            mov_obs_pos = mov_obs_pos.at[:, :2].set(jnp.array([self.area_size, self.area_size])) * 100
            
        mov_obs_vel_key, key = jr.split(key, 2)
        slow_mov_obs_vel = jr.uniform(mov_obs_vel_key, (n_rng_mov_obs // 2, 2), minval=-self._params["slow_mov_obs_speed"], maxval=self._params["slow_mov_obs_speed"])
        # self.slow_mov_obs_vel = slow_mov_obs_vel
        
        # slow_mov_obs_vel = slow_mov_obs_vel.at[:, 2:].set(0)
        
        mov_obs_pos = mov_obs_pos.at[:n_rng_mov_obs // 2, 2:].set(slow_mov_obs_vel)
        
        
        mov_obs_vel_key, key = jr.split(key, 2)
        fast_mov_obs_vel = jr.uniform(mov_obs_vel_key, (n_rng_mov_obs // 2, 2), minval=-self._params["fast_mov_obs_speed"], maxval=self._params["fast_mov_obs_speed"])
        # self.fast_mov_obs_vel = fast_mov_obs_vel
        
        mov_obs_pos = mov_obs_pos.at[n_rng_mov_obs // 2:, 2:].set(fast_mov_obs_vel)
        # fast_mov_obs_vel = fast_mov_obs_vel.at[:, 2:].set(0)
        
        # self.mov_obs_vel = jnp.concatenate([slow_mov_obs_vel, fast_mov_obs_vel], axis=0)
        
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel, mov_obs_pos)

        # add random heading
        theta_key, key = jr.split(key, 2)
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)
        states = states.at[:, 2].set(jr.uniform(theta_key, (self.num_agents,), minval=-np.pi, maxval=np.pi))
        goals = goals.at[:, 2].set(jnp.arctan2(goals[:, 1] - states[:, 1], goals[:, 0] - states[:, 0]))

        env_states = self.EnvState(states, goals, obstacles, mov_obs_pos)

        return self.get_graph(env_states)

    def agent_step_euler(self, agent_states: AgentState, action: Action, stop_mask: Array) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action) * (1 - stop_mask)[:, None]
        n_state_agent_new = agent_states + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([
            (jnp.cos(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (jnp.sin(agent_states[:, 2]) * agent_states[:, 3])[:, None],
            (action[:, 0] * 20.)[:, None],
            (action[:, 1])[:, None]
        ], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        mov_obstacles = graph.env_states.mov_obs

        mov_obs_max_x_mask = mov_obstacles[:, 0] >= self.area_size
        mov_obs_max_y_mask = mov_obstacles[:, 1] >= self.area_size
        mov_obs_min_x_mask = mov_obstacles[:, 0] <= 0
        mov_obs_min_y_mask = mov_obstacles[:, 1] <= 0
        
        mov_obstacles = mov_obstacles.at[:, 2].set(jnp.where(mov_obs_max_x_mask, -mov_obstacles[:, 2], mov_obstacles[:, 2]))
        mov_obstacles = mov_obstacles.at[:, 3].set(jnp.where(mov_obs_max_y_mask, -mov_obstacles[:, 3], mov_obstacles[:, 3]))
        mov_obstacles = mov_obstacles.at[:, 2].set(jnp.where(mov_obs_min_x_mask, -mov_obstacles[:, 2], mov_obstacles[:, 2]))
        mov_obstacles = mov_obstacles.at[:, 3].set(jnp.where(mov_obs_min_y_mask, -mov_obstacles[:, 3], mov_obstacles[:, 3]))
        
        next_mov_obstacles = mov_obstacles.at[:, :2].set(mov_obstacles[:, :2] + mov_obstacles[:, 2:] * self.dt)

        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        if not self.enable_stop:
            # If stopping is not enabled, then set stop_mask to always be 0.
            stop_mask = 0 * stop_mask
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goal_states, obstacles,next_mov_obstacles)

        return self.get_graph(next_state), reward, cost, done, {}

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        mov_obstacles = graph.env_states.mov_obs[:, :2]


        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()
        
        # collision between agents and slow moving obstacles
        collision_mov_obs = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(mov_obstacles[:self._params["n_mov_obs"] // 2], 0), axis=-1)
        collision = (self._params["car_radius"] *2 > collision_mov_obs).any(axis=1)
        cost += collision.mean()

        # collision between agents and fast moving obstacles
        collision_mov_obs = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(mov_obstacles[self._params["n_mov_obs"] // 2:], 0), axis=-1)
        collision = (self._params["car_radius"]*2 > collision_mov_obs).any(axis=1)
        cost += collision.mean()

        return cost

    def render_video(
        rollout, video_path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            n_mov_obs=self.params["n_mov_obs"],
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State):
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        pos_theta_diff = state.agent[:, None, :2] - state.agent[None, :, :2]
        agent_v = jnp.concatenate([(state.agent[:, 3] * jnp.cos(state.agent[:, 2]))[:, None],
                                   (state.agent[:, 3] * jnp.sin(state.agent[:, 2]))[:, None]], axis=-1)
        v_diff = agent_v[:, None, :] - agent_v[None, :, :]
        state_diff = jnp.concatenate([pos_theta_diff, v_diff], axis=-1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        agent_goal_pos_diff = state.agent[:, :2] - state.goal[:, :2]
        agent_goal_v_diff = agent_v
        agent_goal_edge_feats = jnp.concatenate([agent_goal_pos_diff, agent_goal_v_diff], axis=-1)
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_edge_feats = agent_goal_edge_feats.at[:, :2].set(agent_goal_edge_feats[:, :2] * coef)
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        for i in range(self.num_agents):
            agent_goal_edges.append(
                EdgeBlock(agent_goal_edge_feats[i][None, None, :], jnp.ones((1, 1)), id_agent[i][None], id_goal[i][None]))

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = jnp.concatenate([state.agent[i, :2], agent_v[i]]) - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        # agent - mov obs connection
        id_mov_obs = jnp.arange(self.num_agents * 2 + n_hits, self.num_agents * 2 + n_hits + self._params["n_mov_obs"])
        agent_mov_obs_edges = []
        mov_obs_pos = state.mov_obs[:, :2]
        # mov_obs_vel = self.mov_obs_vel_pred(state=state.mov_obs)
        # mov_obs_vel = self.mov_obs_vel[:, :2]
        mov_obs_vel  = state.mov_obs[:, 2:]
        agent_mov_obs_pos = agent_pos[:, None, :2] - mov_obs_pos[None, :, :]
        agent_mov_obs_dist = jnp.linalg.norm(agent_mov_obs_pos, axis=-1)
        agent_mov_obs_mask = jnp.less(agent_mov_obs_dist, self._params["comm_radius"])
        agent_mov_obs_vel = agent_v[:, None, :] - mov_obs_vel[None, :, :]
        agent_mov_obs_edges = EdgeBlock(jnp.concatenate([agent_mov_obs_pos, agent_mov_obs_vel], axis=-1), agent_mov_obs_mask, id_agent, id_mov_obs)
        # for i in range(self.num_agents):
        #     # id_mov_obs = jnp.arange(self.num_agents * 2 + n_hits, self.num_agents * 2 + n_hits + self._params["n_mov_obs"])
        #     agent_mov_obs_pos = agent_pos[i, :2] - mov_obs_pos
        #     agent_mov_obs_dist = jnp.linalg.norm(agent_mov_obs_pos, axis=-1)
        #     agent_mov_obs_mask = jnp.ones((1, self._params["n_mov_obs"]))
        #     mov_obs_rel_vel = agent_v[i] - mov_obs_vel
        #     agent_mov_obs_pos = jnp.concatenate([agent_mov_obs_pos, mov_obs_rel_vel], axis=-1)
        #     # agent_mov_obs_pos = jnp.concatenate([agent_mov_obs_pos, jnp.ones((self._params["n_mov_obs"], 1)) * self._params["mov_obs_speed"], jnp.ones((self._params["n_mov_obs"], 1)) * self._params["car_radius"] * 4], axis=-1)
        #     agent_mov_obs_mask = jnp.logical_and(jnp.less(agent_mov_obs_dist, self._params["comm_radius"]), agent_mov_obs_mask)
        #     agent_mov_obs_edges.append(
        #         EdgeBlock(agent_mov_obs_pos[None, :, :], agent_mov_obs_mask, id_agent[i][None], id_mov_obs)
        #     )
        
        
        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges + [agent_mov_obs_edges]

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([
            (jnp.cos(state[:, 2]) * state[:, 3])[:, None],
            (jnp.sin(state[:, 2]) * state[:, 3])[:, None],
            jnp.zeros((state.shape[0], 2))
        ], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.array([[10., 0.], [0., 1.]])], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], 4, 2)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        v = jnp.concatenate([(state[:, 3] * jnp.cos(state[:, 2]))[:, None],
                             (state[:, 3] * jnp.sin(state[:, 2]))[:, None]], axis=-1)
        edge_state = jnp.concatenate([state[:, :2], v], axis=-1)
        assert edge_state.shape[1] == self.edge_dim
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)
        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        num_mov_obs = self._params["n_mov_obs"]
        n_nodes = 2 * self.num_agents + n_hits + num_mov_obs
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits + num_mov_obs, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, -1].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, -2].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits-num_mov_obs:-num_mov_obs, -3].set(1)  # obs feats
        node_feats = node_feats.at[-num_mov_obs:-num_mov_obs//2, max(self.node_dim-4, 0)].set(1)  # slow mov obs feats
        node_feats = node_feats.at[-num_mov_obs//2:, max(self.node_dim-5, 0)].set(1)  # fast mov obs feats
        
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

        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(DubinsCarAdapt.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DubinsCarAdapt.GOAL)
        node_type = node_type.at[-n_hits-self._params["n_mov_obs"]: -self.params["n_mov_obs"]].set(DubinsCarAdapt.OBS)
        node_type = node_type.at[-self._params["n_mov_obs"]:-self._params["n_mov_obs"] // 2].set(DubinsCarAdapt.SLOW_MOV_OBS)
        node_type = node_type.at[-self._params["n_mov_obs"] // 2:].set(DubinsCarAdapt.FAST_MOV_OBS)

        # node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        # node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DubinsCarAdapt.GOAL)
        # node_type = node_type.at[-n_hits-self._params["n_mov_obs"]: -self.params["n_mov_obs"]].set(DubinsCarAdapt.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 2))], axis=-1)
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
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -0.8])
        upper_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.8])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        lower_lim = jnp.ones(2) * -3.0
        upper_lim = jnp.ones(2) * 3.0
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        pos_diff = agent_states[:, :2] - goal_states[:, :2]

        # PID parameters
        k_omega = 1.0  # 0.5
        k_v = 2.3
        k_a = 2.5

        dist = jnp.linalg.norm(pos_diff, axis=-1)
        theta_t = jnp.arctan2(-pos_diff[:, 1], -pos_diff[:, 0]) % (2 * jnp.pi)
        theta = agent_states[:, 2] % (2 * jnp.pi)
        theta_diff = theta_t - theta
        omega = jnp.zeros(agent_states.shape[0])
        agent_dir = jnp.concatenate([jnp.cos(theta)[:, None], jnp.sin(theta)[:, None]], axis=-1)
        assert agent_dir.shape == (agent_states.shape[0], 2)
        theta_between = jnp.arccos(
            jnp.clip(jnp.matmul(-pos_diff[:, None, :], agent_dir[:, :, None]).squeeze() / (dist + 0.0001),
                     a_min=-1, a_max=1))

        # when theta <= pi
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0), theta <= jnp.pi),
                          k_omega * theta_between, omega)
        # clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff < jnp.pi, theta_diff >= 0)), theta <= jnp.pi),
            -k_omega * theta_between, omega
        )

        # when theta > pi
        # clockwise
        omega = jnp.where(jnp.logical_and(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0), theta > jnp.pi),
                          -k_omega * theta_between, omega)
        # anti-clockwise
        omega = jnp.where(jnp.logical_and(
            jnp.logical_not(jnp.logical_and(theta_diff > -jnp.pi, theta_diff <= 0)), theta > jnp.pi),
            k_omega * theta_between, omega
        )

        omega = jnp.clip(omega, a_min=-5., a_max=5.)

        pos_diff_norm = jnp.sqrt(1e-6 + jnp.sum(pos_diff ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(pos_diff_norm, comm_radius)
        coef = jnp.where(pos_diff_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        pos_diff = coef * pos_diff
        a = -k_a * agent_states[:, 3] + k_v * jnp.linalg.norm(pos_diff, axis=-1)

        action = jnp.concatenate([omega[:, None], a[:, None]], axis=-1)
        return action

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        stop_mask = self.stop_mask(graph)
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )
        
        safe_mask = jnp.logical_and(safe_agent, safe_obs)
        
        safe_mov_mask = self.mov_obs_safe_mask(graph)
        
        safe_mask = jnp.logical_and(safe_mask, safe_mov_mask)

        return safe_mask

    def mov_obs_safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        mov_obs_pos = graph.env_states.mov_obs[:, :2]
        
        slow_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        slow_mov_obs_mask = jnp.greater(slow_mov_obs_dist, self.safe_slow_obs_dist + self._params["car_radius"] * 0.5)

        fast_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        fast_mov_obs_mask = jnp.greater(fast_mov_obs_dist, self.safe_fast_obs_dist + self._params["car_radius"] * 1.0)
        
        safe_mov_obs = jnp.logical_and(slow_mov_obs_mask, fast_mov_obs_mask)

        return safe_mov_obs
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        heading_vec = jnp.concatenate([jnp.cos(agent_state[:, 2])[:, None],
                                       jnp.sin(agent_state[:, 2])[:, None]], axis=1)[:, None, :]
        heading_vec = heading_vec.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist ** 2 - 4 * self._params['car_radius'] ** 2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist ** 2 - self._params['car_radius'] ** 2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        collision_mask =  jnp.logical_or(collision_mask, unsafe_dir)
        
        # agents are colliding with slow moving obstacles
        mov_obs_pos = graph.env_states.mov_obs[:, :2]
        # mov_obs_vel = graph.env_states.mov_obs[:, 2:]
        # for _ in range(10):
        slow_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        unsafe_slow_mov_obs = jnp.less(slow_mov_obs_dist, self.safe_slow_obs_dist).any(axis=1)
        
        fast_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        unsafe_fast_mov_obs = jnp.less(fast_mov_obs_dist, self.safe_fast_obs_dist).any(axis=1)
        
        unsafe_obs = jnp.logical_or(unsafe_fast_mov_obs, unsafe_slow_mov_obs)
        
        collision_mask = jnp.logical_or(collision_mask, unsafe_obs)
        # mov_obs_pos = mov_obs_pos + mov_obs_vel * self.dt
            
        return collision_mask

    def unsafe_mov_obs_mask(self, graph: GraphsTuple) -> Array:
        
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]
        
        mov_obs_pos = graph.env_states.mov_obs[:, :2]
        mov_obs_vel = self.mov_obs_vel[:, :2]
        dt = self.dt
        
        slow_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        unsafe_slow_mov_obs = jnp.less(slow_mov_obs_dist, self.safe_slow_obs_dist).any(axis=1)
        unsafe_obs = unsafe_slow_mov_obs
        
        fast_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        unsafe_fast_mov_obs = jnp.less(fast_mov_obs_dist, self.safe_fast_obs_dist).any(axis=1)
        
        unsafe_obs = jnp.logical_or(unsafe_fast_mov_obs, unsafe_obs)
            
        return unsafe_obs
    
    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)
        
        # agents are colliding with slow moving obstacles
        mov_obs_pos = graph.env_states.mov_obs[:, :2]
        slow_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        unsafe_slow_mov_obs = jnp.less(slow_mov_obs_dist, 2 * self._params["car_radius"]).any(axis=1)
        
        # agents are colliding with fast moving obstacles
        fast_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        unsafe_fast_mov_obs = jnp.less(fast_mov_obs_dist, 3 * self._params["car_radius"]).any(axis=1)
        
        unsafe_mask = jnp.logical_or(unsafe_fast_mov_obs, unsafe_slow_mov_obs)
        # unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_slow_mov_obs)
        # unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_fast_mov_obs)
        
        collision_mask = jnp.logical_or(unsafe_mask, collision_mask)
        
        if self.use_stop_mask:
            stop_mask = self.stop_mask(graph)
            collision_mask = jnp.logical_and(collision_mask, jnp.logical_not(stop_mask))

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

    def stop_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        stop = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 0.5
        return stop

    def agent_collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        return unsafe_agent
    
    def obs_collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]


        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        return unsafe_obs
    
    def mov_obs_collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding with slow moving obstacles
        mov_obs_pos = graph.env_states.mov_obs[:, :2]
        slow_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, :self._params["n_mov_obs"] // 2, :]
        slow_mov_obs_dist = jnp.linalg.norm(slow_mov_obs_dist, axis=-1)
        unsafe_slow_mov_obs = jnp.less(slow_mov_obs_dist, self._params["car_radius"] * 2).any(axis=1)
        
        # agents are colliding with fast moving obstacles
        fast_mov_obs_dist = agent_pos[:, None, :2] - mov_obs_pos[None, self._params["n_mov_obs"] // 2:, :]
        fast_mov_obs_dist = jnp.linalg.norm(fast_mov_obs_dist, axis=-1)
        unsafe_fast_mov_obs = jnp.less(fast_mov_obs_dist, self._params["car_radius"] * 2).any(axis=1)
        
        unsafe_mask = jnp.logical_or(unsafe_fast_mov_obs, unsafe_slow_mov_obs)
        # unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_slow_mov_obs)
        # unsafe_mask = jnp.logical_or(unsafe_mask, unsafe_fast_mov_obs)
        stop_mask = self.stop_mask(graph)
        unsafe_mask = jnp.logical_and(unsafe_mask, jnp.logical_not(stop_mask))
        
        return unsafe_mask
