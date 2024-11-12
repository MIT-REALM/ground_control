import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import Optional, Tuple, NamedTuple

from dgppo.cmarl.env.obstacle import Obstacle
from dgppo.cmarl.trainer.data import Rollout
from dgppo.cmarl.utils.graph import EdgeBlock, GraphsTuple, GetGraph
from dgppo.cmarl.utils.typing import Action, Array, Pos2d, Reward, State, Cost, Done, Info
from dgppo.cmarl.env.lidar_env.base import LidarEnvState
from dgppo.cmarl.utils.utils import jax_vmap, merge01, tree_index
from dgppo.cmarl.env.lidar_env.lidar_target import LidarTarget
from dgppo.cmarl.env.utils import get_rectangle_obstacle_rng


class LidarCircleEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle
    move_obs: State

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


LidarCircleGraphsTuple = GraphsTuple[State, LidarCircleEnvState]


class LidarCircle(LidarTarget):
    AGENT = 0
    GOAL = 1
    OBSTACLE = 2
    MOVE_OBS = 3

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "n_move_obs": 2,
        "move_obs_vel": 0.2,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        'R': 0.5,
        'goal_vel': 0.3
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarCircle.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarCircle, self).__init__(num_agents, area_size, max_step, dt, params)

    @property
    def node_dim(self) -> int:
        return 8  # state dim (4) + indicator: agent 0001, goal 0010, obstacle 0100, move_obstacle 1000

    @property
    def n_cost(self) -> int:
        return 3

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "moving obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        # generate agent and goal
        R = self._params['R']
        theta_key, key = jr.split(key)
        thetas = jnp.linspace(0, 2 * jnp.pi, self.num_agents + 1)[:-1]
        thetas += jr.uniform(theta_key, (1,), minval=0, maxval=2 * jnp.pi)
        agent_pos = jnp.stack([R * jnp.cos(thetas), R * jnp.sin(thetas)], axis=-1) + self.area_size / 2
        states = jnp.concatenate([agent_pos, jnp.zeros((self.num_agents, 2))], axis=-1)
        goals = states.copy()
        goal_vel = jnp.stack([-self._params['goal_vel'] * jnp.sin(thetas),
                              self._params['goal_vel'] * jnp.cos(thetas)], axis=-1)
        goals = goals.at[:, 2:].set(goal_vel)

        # generate obstacles
        if self.params['n_obs'] == 0:
            obstacles = None
        else:
            obs_key, key = jr.split(key, 2)
            obstacles = get_rectangle_obstacle_rng(
                obs_key, self.area_size, self._params['n_obs'], self._params['obs_len_range'],
                self._params['car_radius'] * 1.1, jnp.concatenate([agent_pos, goals[:, :2]], axis=0)
            )

        # generate moving obstacles
        if self.params['n_move_obs'] == 0:
            move_obstacles = None
        else:
            obs_pos = jnp.zeros((self.params['n_move_obs'], 2)) + self.area_size / 2
            thetas = jnp.linspace(0, 2 * jnp.pi, self.params['n_move_obs'] + 1)[:-1]
            v_dir = jnp.stack([jnp.cos(thetas - jnp.pi / 2), jnp.sin(thetas - jnp.pi / 2)], axis=-1)
            obs_vel = self._params['move_obs_vel'] * v_dir
            move_obstacles = jnp.concatenate([obs_pos, obs_vel], axis=-1)

        env_states = LidarCircleEnvState(states, goals, obstacles, move_obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def step(
            self, graph: LidarCircleGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarCircleGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = graph.env_states.obstacle if self.params['n_obs'] > 0 else None

        # calculate next states
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        lidar_data_next = self.get_lidar_data(next_agent_states, obstacles)
        info = {}

        # calculate next goals
        thetas = jnp.arctan2(goals[:, 1] - self.area_size / 2, goals[:, 0] - self.area_size / 2)
        thetas_next = thetas + self._params['goal_vel'] * self.dt / self._params['R']
        next_goal_pos = jnp.stack([self.area_size / 2 + self._params['R'] * jnp.cos(thetas_next),
                                   self.area_size / 2 + self._params['R'] * jnp.sin(thetas_next)], axis=-1)
        next_goal_vel = jnp.stack([-self._params['goal_vel'] * jnp.sin(thetas_next),
                                   self._params['goal_vel'] * jnp.cos(thetas_next)], axis=-1)
        next_goals = goals.at[:, :2].set(next_goal_pos).at[:, 2:].set(next_goal_vel)

        # calculate next moving obstacles
        move_obs = graph.env_states.move_obs
        center_thetas = jnp.linspace(0, 2 * jnp.pi, self.params['n_move_obs'] + 1)[:-1]
        center_pos = (self._params['R'] * 2 / 3 *
                      jnp.stack([jnp.cos(center_thetas), jnp.sin(center_thetas)], axis=-1) + self.area_size / 2)
        obs_thetas = jnp.arctan2(move_obs[:, 1] - center_pos[:, 1], move_obs[:, 0] - center_pos[:, 0])
        obs_thetas_next = obs_thetas + self._params['move_obs_vel'] * self.dt / (self._params['R'] * 2 / 3)
        next_obs_pos = center_pos + (self._params['R'] * 2 / 3) * jnp.stack([jnp.cos(obs_thetas_next),
                                                                             jnp.sin(obs_thetas_next)], axis=-1)
        next_obs_vel = self._params['move_obs_vel'] * jnp.stack([-jnp.sin(obs_thetas_next),
                                                                 jnp.cos(obs_thetas_next)], axis=-1)
        next_move_obs = jnp.concatenate([next_obs_pos, next_obs_vel], axis=-1)

        next_state = LidarCircleEnvState(next_agent_states, next_goals, obstacles, next_move_obs)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        return self.get_graph(next_state, lidar_data_next), reward, cost, done, info

    def get_reward(self, graph: LidarCircleGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
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

    def get_cost(self, graph: LidarCircleGraphsTuple) -> Cost:
        cost_0 = super(LidarCircle, self).get_cost(graph)

        # collision between agents and moving obstacles
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        move_obs = graph.type_states(type_idx=3, n_type=self.params['n_move_obs'])[:, :2]
        dist = jnp.linalg.norm(agent_pos[:, None, :] - move_obs[None, :, :], axis=-1)
        move_obs_cost: Array = self.params['car_radius'] + self.params['car_radius'] - dist.min(axis=1)

        eps = 0.5
        cost_1 = jnp.where(move_obs_cost <= 0.0, move_obs_cost - eps, move_obs_cost + eps)
        cost_1 = jnp.clip(cost_1, -1.0, 1.0)

        cost = jnp.concatenate([cost_0, cost_1[:, None]], axis=-1)
        return cost

    def get_render_data(self, rollout: Rollout, Ta_is_unsafe: Array = None) -> Tuple[dict, dict]:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        graph0 = tree_index(rollout.graph, 0)
        n_moving_node = self.num_agents + self.num_goals + self.params['n_move_obs']

        T_moving_node_pos = []
        T_edge_index = []
        T_edge_node_pos = []
        T_edge_colors = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)

            # get positions of nodes
            agent_pos = graph_t.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
            goal_pos = graph_t.type_states(type_idx=1, n_type=self.num_goals)[:, :2]
            obs_pos = graph_t.type_states(type_idx=3, n_type=self.params['n_move_obs'])[:, :2]
            T_moving_node_pos.append(jnp.concatenate([goal_pos, obs_pos, agent_pos], axis=0))

            # get edge index
            e_edge_index_t = np.stack([graph_t.senders, graph_t.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t ==
                              self.num_agents + self.num_goals + n_hits + self.params['n_move_obs'], axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            T_edge_index.append(e_edge_index_t)

            # get positions of nodes in edges
            T_edge_node_pos.append(graph_t.states[:, :2])

            # get edge colors
            e_is_goal_t = (self.num_agents <= graph_t.senders) & (graph_t.senders < self.num_agents + self.num_goals)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = ["#2fdd00" if e_is_goal_t[ii] else "0.2" for ii in range(e_edge_index_t.shape[1])]
            T_edge_colors.append(e_colors_t)

        settings = {
            "n_static_node": 0,
            "n_moving_node": n_moving_node,
            "moving_node_r": [self.params["car_radius"]] * n_moving_node,
            "moving_node_color": ["#2fdd00"] * self.num_goals +
                                 ["#ff0000"] * self.params['n_move_obs'] + ["#0068ff"] * self.num_agents,
            "moving_node_labels": [None] * (self.num_goals + self.params['n_move_obs']) +
                                  [f"{i}" for i in range(self.num_agents)],
            "cost_components": self.cost_components,
            "obstacle_color": "#8a0000"
        }
        T_data = {
            "T_moving_node_pos": T_moving_node_pos,
            "Ta_is_unsafe": Ta_is_unsafe,
            "T_costs": rollout.costs,
            "T_rewards": rollout.rewards,
            "T_edge_index": T_edge_index,
            "T_edge_node_pos": T_edge_node_pos,
            "T_edge_colors": T_edge_colors,
            "static_obstacles": graph0.env_states.obstacle
        }
        return settings, T_data

    def edge_blocks(self, state: LidarCircleEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        lidar_target_env_state = LidarEnvState(state.agent, state.goal, state.obstacle)
        edge_blocks = super(LidarCircle, self).edge_blocks(lidar_target_env_state, lidar_data)

        # agent - moving obstacle connection
        agent_pos = state.agent[:, :2]
        move_obs = state.move_obs[:, :2]
        pos_diff = agent_pos[:, None, :] - move_obs[None, :, :]
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.move_obs)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        agent_obs_mask = jnp.less(dist, self._params['comm_radius'])
        id_agent = jnp.arange(self.num_agents)
        n_hits = self._params["top_k_rays"] * self.num_agents
        id_obs = self.num_agents + self.num_goals + n_hits + jnp.arange(self.params['n_move_obs'])
        agent_obs_edges = EdgeBlock(edge_feats, agent_obs_mask, id_agent, id_obs)

        edge_blocks.append(agent_obs_edges)
        return edge_blocks

    def get_graph(self, state: LidarCircleEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        n_move = self.params['n_move_obs']
        n_nodes = self.num_agents + self.num_goals + n_hits + n_move

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + n_hits + n_move, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(
            state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[self.num_agents + self.num_goals:
                                       self.num_agents + self.num_goals + n_hits, :2].set(lidar_data)
        if n_move > 0:
            node_feats = node_feats.at[-n_move:, :self.state_dim].set(state.move_obs)

        # indicators
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 3].set(1.)  # agent
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 2].set(1.))  # goal
        if n_hits > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:
                                       self.num_agents + self.num_goals + n_hits, self.state_dim + 1].set(1.)
        if n_move > 0:
            node_feats = node_feats.at[-n_move:, self.state_dim].set(1.)

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(LidarCircle.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(LidarCircle.GOAL)
        if n_hits > 0:
            node_type = node_type.at[self.num_agents + self.num_goals:
                                     self.num_agents + self.num_goals + n_hits].set(LidarCircle.OBS)
        if n_move > 0:
            node_type = node_type.at[-n_move:].set(LidarCircle.MOVE_OBS)

        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]))], axis=1)
            states = jnp.concatenate([states, lidar_states], axis=0)
        if n_move > 0:
            states = jnp.concatenate([states, state.move_obs], axis=0)
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states
        ).to_padded()