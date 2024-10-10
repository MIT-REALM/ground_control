import functools as ft
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import pytictoc

from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.policy import ControlPolicy

from cmarl.cmarl.algo import make_algo, EFInforMARL
from cmarl.cmarl.env import make_env
from cmarl.cmarl.env.lidar_env import LidarEnvState
from cmarl.cmarl.trainer.utils import get_bb_cbf, plot_rnn_states, test_rollout, get_bb_Vh
from cmarl.cmarl.utils.graph import GraphsTuple
from cmarl.cmarl.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap, np2jax, jax2np

t = pytictoc.TicToc()
# from policies.common import F1TenthAction
# from policies.policy import ControlPolicy

# from . import F1TenthAction
# from common import F1TenthAction
# from policy import ControlPolicy

class CMARL_policy(ControlPolicy):
    def __init__(
            self, 
            min_distance: float = 1.0,
            car_pos: np.ndarray = np.array([3.0, -0.5, 0.0, 0,0]),
            car_goal: np.ndarray = np.array([1.0, 0.5, 0.0, 0.0]),
            obs_pos: np.ndarray = np.zeros((20, 2)),
            num_obs: int = 2,
            mov_obs: int = 20,
            model_path = 'realm_gc/rgc_control/src/cmarl/logs/LidarF1TenthTarget'
            ):
        self.min_distance = min_distance
        car_pos = car_pos
        car_goal = car_goal
        obs_pos = obs_pos
        num_obs=num_obs
        mov_obs=mov_obs
        mov_obs_speed=0.1
        model_path = model_path
        with open(os.path.join(model_path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        num_agents = 1
        node_dim = config.node_feat if "node_feat" in config else 3
        env = make_env(
            env_id=config.env,
            num_agents=num_agents,
            num_obs=mov_obs,
            max_step=128,
            max_travel=100,
            full_observation=True,
            n_mov_obs=mov_obs,
            delta_scale=10.0,
            goal_reward_scale=config.goal_reward_scale if "goal_reward_scale" in config else 1.0,
        )
        
        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            cost_weight=config.cost_weight,
            actor_gnn_layers=config.actor_gnn_layers,
            Vl_gnn_layers=config.Vl_gnn_layers,
            Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
            lr_actor=config.lr_actor,
            lr_Vl=config.lr_Vl,
            max_grad_norm=2.0,
            seed=config.seed,
            use_rnn=config.use_rnn,
            rnn_layers=config.rnn_layers,
            use_lstm=config.use_lstm,
        )
        model_path = os.path.join(model_path, "models")
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        algo.load(model_path, step)

        act_fn = jax.jit(algo.act)
        self.init_rnn_state = algo.init_rnn_state
        # params = algo.cbf_train_state.params
        self.act_fn = act_fn
        self.step = jax.jit(env.step)

        self.Vh_fn = jax.jit(algo.get_Vh)

        key=jax.random.PRNGKey(0)
        graph0 = env.reset(key)
        _ = self.act_fn(graph0, self.init_rnn_state)
        self.env = env
        self.graph0 = graph0
        self.car_goal = car_goal
        self.graph0 = self.create_graph(car_pos, car_goal, obs_pos)
        self.obs = obs_pos
        
    def create_graph(self, car_pos, car_goal, obs_pos, graph=None):
        
        if graph is None:
            graph = self.graph0
        states=graph.env_states
        obs = states.obstacle
        agent_states= car_pos[:2]
        theta = car_pos[None, 2]
        v = car_pos[None, -1]
        agent_states = jnp.concatenate([agent_states, jnp.cos(theta), jnp.sin(theta), v])[None, :]

        goal_states = car_goal[:2]
        goal_theta = car_goal[None,2]
        goal_v = car_goal[None,-1]
        goal_states = jnp.concatenate([goal_states, jnp.cos(goal_theta), jnp.sin(goal_theta), goal_v])[None, :]

        mov_obs = obs_pos
        if graph is not None:
            mov_obs_vel = self.env.mov_obs_vel_pred(graph)
        else:
            # mov_obs_vel = self.env.mov_agent_vel_pred(state=obs_pos)
            mov_obs_vel = jnp.zeros((mov_obs.shape[0], 2))
        
        # mov_obs = jnp.concatenate([obs_pos[:,0], obs_pos[:,1], jnp.zeros(obs_pos.shape[0]), jnp.zeros(obs_pos.shape[0])], axis=0)
        # print('mov obs vel shape:', mov_obs_vel.shape)
        # print('mov_obs shape:', mov_obs.shape)
        mov_obs = jnp.concatenate([mov_obs, mov_obs_vel, jnp.zeros((mov_obs.shape[0], 1))], axis=1)
        states = LidarEnvState(agent=agent_states, goal=goal_states, obstacle=obs, mov_obs=mov_obs)

        graph = self.env.get_graph(states)
        return graph
        
    def compute_action(
        self,
        car_pos,
        ref_inp=None, 
        obs=None,
        mov_obs_vel=None,
        goal=None,
        dt=0.01,
    ) -> F1TenthAction:
        # Brake to avoid collisions based on the average distance to the
        # obstacle in the center of the image
        graph = self.graph0
        if obs is None:
            obs = self.obs
        if goal is None:
            goal = self.car_goal
        
        car_pos = jnp.array([car_pos.x, car_pos.y, car_pos.theta, car_pos.v])
        
        new_graph = self.create_graph(car_pos, goal, obs, graph)
        self.graph0 = new_graph

        Vh = self.Vh_fn(graph, self.init_rnn_state)
        next_ref_graph = self.step(graph, jnp.array([ref_inp.steering_angle, ref_inp.acceleration])[None, :])
        Vh_next = self.Vh_fn(next_ref_graph, self.init_rnn_state)
        Vh_dot = (Vh_next - Vh) / dt

        Vhcond = Vh_dot + 10 * Vh
        max_Vhcond = jnp.max(Vhcond)

        print('Vh cond max: ', max_Vhcond)
        print('vh: ', Vh)
        if max_Vhcond < 0:
            return ref_inp, next_ref_graph.env_states.agent.squeeze(), 1
        # print('graph state before step: ', new_graph.env_states.agent)
        # if ref_inp is not None:
        #     ref_vel = jnp.array([ref_inp.steering_angle, ref_inp.acceleration])
        # else:
        #     ref_vel = None

        # mov_obs_vel = self.env.mov_obs_vel_pred(new_graph)
        # t.tic()
        num_iter = 20
        states = jnp.zeros((num_iter, 5))
        for j in range(num_iter):
            accel, self.init_rnn_state = self.act_fn(new_graph, self.init_rnn_state)
            accel = self.env.clip_action(accel)
            # print('action time: ', t.tocvalue())
            # t.tic()

            new_graph = self.step(new_graph, accel)
            # print('state shape: ', new_graph.env_states.agent.shape)
            states = states.at[j].set(new_graph.env_states.agent.squeeze())
        # print('step time: ', t.tocvalue())
        # obs_coll = self.env.mov_obs_collision_mask(new_graph)
        # print('obs coll:', obs_coll * 1)
        
        # next_state = new_graph.env_states.agent
        next_state = states
        # print('accel: ', accel)
        # print('graph state after step: ', new_graph.env_states.agent)

        return F1TenthAction(
            acceleration=accel[0, 1],
            steering_angle=accel[0, 0],
        ), next_state.squeeze(), 0
 