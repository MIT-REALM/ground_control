import jax.numpy as jnp
import jax.random as jr
import jax

from typing import Tuple, Optional

from dgppo.cmarl.env.utils import get_rectangle_obstacle_rng
from dgppo.cmarl.trainer.data import Rollout
from dgppo.cmarl.utils.graph import GraphsTuple
from dgppo.cmarl.utils.typing import Action, Array, State, AgentState, Reward, Cost, Done, Info
from dgppo.cmarl.env.lidar_env.lidar_circle import LidarCircle, LidarCircleEnvState, LidarCircleGraphsTuple
from dgppo.cmarl.utils.utils import tree_index


class LidarBicycleCircle(LidarCircle):

    PARAMS = {
        "car_radius": 0.2,
        "comm_radius": 3,
        "n_rays": 32,
        "obs_len_range": [0.4, 1.2],
        "n_obs": 3,
        "n_move_obs": 3,
        "move_obs_vel": 0.8,
        "default_area_size": 7,
        "dist2goal": 0.04,
        "top_k_rays": 8,
        'R': 2.5,
        'goal_vel': 1.0,
        'L': 0.28,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarBicycleCircle.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarBicycleCircle, self).__init__(num_agents, area_size, max_step, dt, params)

    @property
    def state_dim(self) -> int:
        return 5  # x, y, cos(theta), sin(theta), v

    @property
    def node_dim(self) -> int:
        return 9  # state dim (4) + indicator: agent 0001, goal 0010, obstacle 0100, move_obstacle 1000

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    def reset(self, key: Array) -> GraphsTuple:
        # generate agent and goal
        R = self._params['R']
        theta_key, key = jr.split(key)
        thetas = jnp.linspace(0, 2 * jnp.pi, self.num_agents + 1)[:-1]
        thetas += jr.uniform(theta_key, (1,), minval=0, maxval=2 * jnp.pi)
        agent_pos = jnp.stack([R * jnp.cos(thetas), R * jnp.sin(thetas)], axis=-1) + self.area_size / 2
        states = jnp.concatenate([agent_pos, jnp.zeros((self.num_agents, 3))], axis=-1)
        goals = states.copy()
        v_dir = jnp.stack([-jnp.sin(thetas), jnp.cos(thetas)], axis=-1)
        states = states.at[:, 2:4].set(v_dir)
        goals = goals.at[:, 2:4].set(v_dir)
        goals = goals.at[:, 4].set(self._params['goal_vel'])

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
            def get_obs_(inp):
                this_key, _, center_theta_ = inp
                theta_key_, this_key = jr.split(this_key, 2)
                theta_ = jr.uniform(theta_key_, (), minval=0, maxval=2 * jnp.pi)
                # center_theta_ = center_thetas[i_obs]
                center_ = (self._params['R'] * 2 / 3 * jnp.array([jnp.cos(center_theta_), jnp.sin(center_theta_)])
                           + self.area_size / 2)  # center of the trajectory of the moving obstacle
                assert center_.shape == (2,)
                pos_ = center_ + (self._params['R'] * 2 / 3) * jnp.array([jnp.cos(theta_), jnp.sin(theta_)])
                assert pos_.shape == (2,)
                return this_key, pos_, center_theta_

            def non_valid_obs_(inp):
                _, pos_, _ = inp
                dist_min_agents = jnp.linalg.norm(agent_pos - pos_, axis=1).min()
                return dist_min_agents < self._params["car_radius"] * 2

            def get_valid_obs_(carry, inp):
                this_key, center_theta_ = inp
                theta_key_, this_key = jr.split(this_key, 2)
                theta_ = jr.uniform(theta_key_, (), minval=0, maxval=2 * jnp.pi)
                # center_theta_ = center_thetas[0]
                center_ = (self._params['R'] * 2 / 3 * jnp.array([jnp.cos(center_theta_), jnp.sin(center_theta_)])
                           + self.area_size / 2)  # center of the trajectory of the moving obstacle
                assert center_.shape == (2,)
                pos_ = center_ + (self._params['R'] * 2 / 3) * jnp.array([jnp.cos(theta_), jnp.sin(theta_)])
                assert pos_.shape == (2,)
                _, valid_obs_, _ = jax.lax.while_loop(
                    non_valid_obs_, get_obs_, (this_key, pos_, center_theta_))
                return carry, valid_obs_

            obs_key, key = jr.split(key, 2)
            obs_key = jr.split(obs_key, self.params['n_move_obs'])
            center_thetas = jnp.linspace(0, 2 * jnp.pi, self.params['n_move_obs'] + 1)[:-1]
            _, obs_pos = jax.lax.scan(get_valid_obs_, None, (obs_key, center_thetas))
            v_dir = jnp.stack([jnp.cos(center_thetas - jnp.pi / 2), jnp.sin(center_thetas - jnp.pi / 2)], axis=-1)
            obs_vel = jnp.ones((self.params['n_move_obs'], 1)) * self._params['move_obs_vel']
            move_obstacles = jnp.concatenate([obs_pos, v_dir, obs_vel], axis=-1)

        env_states = LidarCircleEnvState(states, goals, obstacles, move_obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        def single_agent_step(x, u):
            theta = jnp.arctan2(x[3], x[2])
            theta_next = theta + x[4] * jnp.tan(u[0] * jnp.deg2rad(20)) * self.dt / self.params['L']
            x_next = jnp.array([
                x[0] + x[4] * jnp.cos(theta) * self.dt,
                x[1] + x[4] * jnp.sin(theta) * self.dt,
                jnp.cos(theta_next),
                jnp.sin(theta_next),
                x[4] + u[1] * self.dt * 4.
            ])
            return x_next

        n_state_agent_new = jax.vmap(single_agent_step)(agent_states, action)

        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def state2feat(self, state: State) -> Array:
        vx = state[4] * state[2]
        vy = state[4] * state[3]
        feat = jnp.concatenate([state[:2], vx[None], vy[None]], axis=-1)
        assert feat.shape == (self.edge_dim,)
        return feat

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
        next_goal_vel_dir = jnp.stack([-jnp.sin(thetas_next), jnp.cos(thetas_next)], axis=-1)
        next_goal_vel = jnp.ones((self.num_goals,)) * self._params['goal_vel']
        next_goals = goals.at[:, :2].set(next_goal_pos).at[:, 2:4].set(next_goal_vel_dir).at[:, 4].set(next_goal_vel)

        # calculate next moving obstacles
        move_obs = graph.env_states.move_obs
        center_thetas = jnp.linspace(0, 2 * jnp.pi, self.params['n_move_obs'] + 1)[:-1]
        center_pos = (self._params['R'] * 2 / 3 *
                      jnp.stack([jnp.cos(center_thetas), jnp.sin(center_thetas)], axis=-1) + self.area_size / 2)
        obs_thetas = jnp.arctan2(move_obs[:, 1] - center_pos[:, 1], move_obs[:, 0] - center_pos[:, 0])
        obs_thetas_next = obs_thetas + self._params['move_obs_vel'] * self.dt / (self._params['R'] * 2 / 3)
        next_obs_pos = center_pos + (self._params['R'] * 2 / 3) * jnp.stack([jnp.cos(obs_thetas_next),
                                                                             jnp.sin(obs_thetas_next)], axis=-1)
        next_obs_vel_dir = jnp.stack([-jnp.sin(obs_thetas_next), jnp.cos(obs_thetas_next)], axis=-1)
        next_obs_vel = jnp.ones((self.params['n_move_obs'], 1)) * self._params['move_obs_vel']
        next_move_obs = jnp.concatenate([next_obs_pos, next_obs_vel_dir, next_obs_vel], axis=-1)

        next_state = LidarCircleEnvState(next_agent_states, next_goals, obstacles, next_move_obs)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        return self.get_graph(next_state, lidar_data_next), reward, cost, done, info

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -1, -1, -1.0])
        upper_lim = jnp.array([self.area_size, self.area_size, 1, 1, 1.8])
        return lower_lim, upper_lim

    def get_render_data(self, rollout: Rollout, Ta_is_unsafe: Array = None) -> Tuple[dict, dict]:
        settings, T_data = super(LidarBicycleCircle, self).get_render_data(rollout, Ta_is_unsafe)

        # add heading and steering angle
        T_heading = []
        T_steering = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)
            agent_states = graph_t.type_states(type_idx=0, n_type=self.num_agents)
            thetas = jnp.arctan2(agent_states[:, 3], agent_states[:, 2])
            invalid_thetas = jnp.full((self.num_goals + self.params['n_move_obs'],), jnp.nan)
            T_heading.append(jnp.concatenate([invalid_thetas, thetas]))
            T_steering.append(jnp.concatenate([invalid_thetas, rollout.actions[kk, :, 0] * jnp.deg2rad(20) + thetas]))
        T_data["T_heading"] = T_heading
        T_data["T_steering"] = T_steering
        settings["heading_color"] = "#ffed47"
        settings["steering_color"] = "#ffffff"

        return settings, T_data