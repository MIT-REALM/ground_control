import jax.numpy as jnp
import numpy as np
import jax

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from .lidar_env import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from ..utils.utils import jax_vmap


class LidarNav(LidarEnv):

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
            params: dict = None,
            goal_reward_scale = 1.0,
    ):
        area_size = LidarNav.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarNav, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.goal_reward_scale = goal_reward_scale
        
    @property
    def reward_min(self) -> float:
        return -((self.area_size * np.sqrt(2)) * 0.01 - 0.001 - 0.0001) * self.max_episode_steps

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.env_states.agent
        goals = graph.env_states.goal
        reward = jnp.zeros(()).astype(jnp.float32)

        # goal distance penalty
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
        reward -= (dist2goal.mean()) * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001 * self.goal_reward_scale

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward

    def state2feat(self, state: State) -> Array:
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            agent_goal_feats_i = self.state2feat(agent_state_i) - self.state2feat(goal_state_i)
            agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))

        # agent - obs connection
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )
        
        # agent - mov obs connection
        n_mov_obs = self._params["n_mov_obs"]
        id_mov_obs = jnp.arange(self.num_agents + self.num_goals + n_hits, self.num_agents + self.num_goals + n_hits + n_mov_obs)
        
        agent_mov_obs_edges = []
        mov_obs_pos = state.mov_obs[:, :2]
        mov_obs_vel = state.mov_obs[:, 2:4]
        agent_mov_obs_pos = agent_pos[:, None, :] - mov_obs_pos[None, :, :]
        agent_feat = jax_vmap(self.state2feat)(state.agent)
        agent_vel = agent_feat[:, 2:4]
        agent_mov_obs_vel = agent_vel[:, None, :] - mov_obs_vel[None, :, :]
        agent_mov_obs_dist = jnp.linalg.norm(agent_mov_obs_pos, axis=-1)
        agent_mos_obs_mask = jnp.less(agent_mov_obs_dist, self._params["comm_radius"])
        mov_obs_feats = jnp.concatenate([agent_mov_obs_pos, agent_mov_obs_vel], axis=-1)
        agent_mov_obs_edges = EdgeBlock(mov_obs_feats, agent_mos_obs_mask, id_agent, id_mov_obs)
                
        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges + [agent_mov_obs_edges]
