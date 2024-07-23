import functools as ft
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env

from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.policy import ControlPolicy

class GCBF_policy(ControlPolicy):
    def __init__(
            self, 
            min_distance: float = 1.0,
            car_pos: np.ndarray = np.array([0.0, 0.0]),
            car_goal: np.ndarray = np.array([1.0, 1.0]),
            obs_pos: np.ndarray = np.array([0.5, 0.5]),
            num_obs: int = 1,
            mov_obs: int = 1,
            model_path = 'realm_gc/rgc_control/src/logs/DubinsCarAdapt/gcbf+/seed1_20240719162242/'
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
            num_obs=num_obs,
            num_mov_obs=mov_obs,
            mov_obs_speed=max(config.mov_obs_speed,mov_obs_speed),
            mov_obs_at_infty=False if not 'mov_obs_at_infty' in config else config.mov_obs_at_infty,
            station_obs_at_infty=False if 'station_obs_at_infty' not in config else config.station_obs_at_infty,
            area_size=4,
            max_step=1024,
            max_travel=10,
            node_feat=node_dim,
            use_stop_mask=True,
        )
        step=1000
        if "dim_factor" not in config:
            dim_factor = 2
        else:
            dim_factor = config.dim_factor
        algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                dim_factor=dim_factor,
                n_agents=env.num_agents,
                gnn_layers=config.gnn_layers,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                horizon=config.horizon,
                lr_actor=config.lr_actor,
                lr_cbf=config.lr_cbf,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed,
            )
        algo.load(model_path, step)
        act_fn = jax.jit(algo.act)
        params = algo.cbf_train_state.params
        # qp_fn = jax.jit(ft.partial(algo.get_u_qp_act,params=params))
        qp_fn = jax.jit(ft.partial(algo.get_u_qp_act, params=params, act=act_fn))
        self.act_fn = qp_fn
        # self.rollout_fn = jax_jit_np(env.rollout_qp_fn(act_fn, 1))
        key=jax.random.PRNGKey(0)
        graph0 = env.reset(key)
        self.env = env
        self.graph0 = graph0
        self.car_goal = car_goal
        self.graph0 = self.init_graph(car_pos, car_goal, obs_pos)
        
    def init_graph(self, car_pos, car_goal, obs_pos, graph=None):
        if graph is None:
            graph = self.graph0
        states=graph.states
        states.agent = jnp.array([car_pos[0], car_pos[1], 0.0, 0.0])
        states.goal = jnp.array([car_goal[0], car_goal[1], 0.0, 0.0])
        states.mov_obs = jnp.cat([obs_pos[:,0], obs_pos[:,1], jnp.zeros(obs_pos.shape[0]), jnp.zeros(obs_pos.shape[0])], axis=0)
        graph = self.env.get_graph(states)
        return graph
        
    def compute_action(
        self,
        car_pos, 
        obs,
        mov_obs_vel=None,
    ) -> F1TenthAction:
        # Brake to avoid collisions based on the average distance to the
        # obstacle in the center of the image
        graph = self.graph0
        new_graph = self.init_graph(car_pos, self.car_goal, obs)
        self.graph0 = new_graph
        accel = self.act_fn(new_graph, graph, mov_obs_vel=mov_obs_vel)
        accel = self.env.clip_action(accel)
        return F1TenthAction(
            acceleration=accel,
            steering_angle=0.0,
        )
    
 