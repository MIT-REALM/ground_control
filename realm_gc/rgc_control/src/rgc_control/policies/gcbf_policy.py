import functools as ft
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from pytictoc import TicToc
from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.policy import ControlPolicy

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env

t = TicToc()

# from policies.common import F1TenthAction
# from policies.policy import ControlPolicy

# from . import F1TenthAction
# from common import F1TenthAction
# from policy import ControlPolicy

class GCBF_policy(ControlPolicy):
    def __init__(
            self, 
            min_distance: float = 1.0,
            car_pos: np.ndarray = np.array([3.0, -0.5, 0.0, 0,0]),
            car_goal: np.ndarray = np.array([1.0, 0.5, 0.0, 0.0]),
            obs_pos: np.ndarray = np.zeros((20, 2)),
            num_obs: int = 20,
            mov_obs: int = 2,
            model_path = 'realm_gc/rgc_control/src/gcbfplus/seed1_20240719162242/'
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
            mov_obs_speed=0.0, #max(config.mov_obs_speed,mov_obs_speed),
            mov_obs_at_infty=False, # if not 'mov_obs_at_infty' in config else config.mov_obs_at_infty,
            station_obs_at_infty=True, #if 'station_obs_at_infty' not in config else config.station_obs_at_infty,
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
        algo.load(os.path.join(model_path, 'models'), step)
        act_fn = jax.jit(algo.act)
        params = algo.cbf_train_state.params
        qp_fn = jax.jit(ft.partial(algo.get_u_qp_act, params=params, act=act_fn))
        self.ref_check_fn = jax.jit(ft.partial(algo.ref_check, params=params, act=act_fn))
        self.act_fn = act_fn
        self.qp_act_fn = qp_fn
        key=jax.random.PRNGKey(0)
        graph0 = env.reset(key)
        _ = self.qp_act_fn(graph0, graph0)
        self.env = env
        self.graph0 = graph0
        self.car_goal = car_goal
        self.graph0 = self.create_graph(car_pos, car_goal, obs_pos)
        self.obs = obs_pos
        self.jit_step = jax.jit(env.step)
        self.jit_rollout = jax.jit(self.rollout)
        graph, _, _ = self.jit_rollout(graph0, graph0, None, None)
        
    def create_graph(self, car_pos, car_goal, obs_pos, graph=None, mov_obs_vel=None):
        if graph is None:
            graph = self.graph0
        states=graph.env_states
        obs = states.obstacle
        agent_states= car_pos[None, :]
        goal_states = car_goal[None, :]
        # mov_obs = jnp.concatenate([obs_pos[:,0], obs_pos[:,1], jnp.zeros(obs_pos.shape[0]), jnp.zeros(obs_pos.shape[0])], axis=0)
        mov_obs = obs_pos
        if mov_obs_vel is not None:
            mov_obs = jnp.concatenate([mov_obs, jnp.array(mov_obs_vel)], axis=1)
        else:
            mov_obs = jnp.concatenate([mov_obs, jnp.zeros((mov_obs.shape[0], 2))], axis=1)
        states = self.env.EnvState(agent=agent_states, goal=goal_states, obstacle=obs, mov_obs=mov_obs)

        # print(type(states))
        # print(type(states.agent))
        # states.agent = car_pos
        # states.goal = car_goal
        # print(obs_pos.shape)
        # breakpoint()
        graph = self.env.get_graph(states)
        return graph
        
    def compute_action(
        self,
        car_pos,
        ref_inp=None, 
        obs=None,
        mov_obs_vel=None,
        goal=None,
    ) -> F1TenthAction:
        # Brake to avoid collisions based on the average distance to the
        # obstacle in the center of the image
        iter_max = 1
        
        
        graph = self.graph0
        if obs is None:
            obs = self.obs
        if goal is None:
            goal = self.car_goal
        
        car_pos = jnp.array([car_pos.x, car_pos.y, car_pos.theta, car_pos.v])
        
        new_graph = self.create_graph(car_pos, goal, obs, graph)
        self.graph0 = new_graph
            # print('graph state before step: ', new_graph.env_states.agent)
        if ref_inp is not None:
            ref_vel = jnp.array([ref_inp.steering_angle, ref_inp.acceleration])[None, :]
            next_graph, _, _, _, _ = self.env.step(new_graph, ref_vel)
            obs_coll = self.env.mov_obs_collision_mask(next_graph)
            flag = 1- max(obs_coll * 1)
            ref_accel = ref_vel
        else:
            ref_vel = None
            flag = 0
                
        mov_obs_vel = self.env.mov_obs_vel_pred(new_graph)
        new_graph = self.create_graph(car_pos, goal, obs, new_graph, mov_obs_vel[:, :2])

        # ref_accel, flag = self.ref_check_fn(new_graph, graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)    
        # if flag == 1:
        accel_ref = ref_accel
        accel_ref = self.env.clip_action(accel_ref)
    
        next_graph, _, _, _, _ = self.env.step(next_graph, accel_ref)
            
        next_state_ref = next_graph.env_states.agent

        # else:  
        # t.tic()
        graph, next_state, accel = self.jit_rollout(new_graph, graph, mov_obs_vel, ref_vel)  
        # time_rollout = t.tocvalue()
        # print('Time for rollout: ', time_rollout)
        # print('new state shape:', next_state.shape)
            # for _ in range(iter_max):
            #     state = graph.env_states.agent
            #     print('prev state: ', state)
                
            #     accel = self.qp_act_fn(new_graph, graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)
            #     # accel = self.act_fn(new_graph)
            #     accel = self.env.clip_action(accel)
                
            #     new_graph, _, _, _, _ = self.jit_step(new_graph, accel)
            #     # new_graph, _, _, _, _ = self.env.step(new_graph, accel)
                
            #     obs_coll = self.env.mov_obs_collision_mask(new_graph)
            #     print('obs coll:', obs_coll * 1)
                
            #     next_state = new_graph.env_states.agent
            #     # print('accel: ', accel)
            #     print('graph state after step: ', new_graph.env_states.agent)
        if flag == 1:
            return F1TenthAction(
                acceleration=accel_ref[0, 1],
                steering_angle=accel_ref[0, 0],
            ), next_state_ref.squeeze(), flag
        else:
            return F1TenthAction(
                acceleration=accel[0, 1],
                steering_angle=accel[0, 0],
            ), next_state.squeeze(), flag
    
    
    def rollout(self, new_graph, graph, mov_obs_vel, ref_vel, iter_max=100):
        
        def body_fn(carry, inp):
            new_graph, graph = carry
            
            accel = self.qp_act_fn(new_graph ,graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)
            accel = self.env.clip_action(accel)
            graph = new_graph
            new_graph,_, _, _, _ = self.env.step(graph, accel)
            new_state = new_graph.env_states.agent
            
            return (new_graph, graph), (new_state, accel)
            
            
            # new_graph, graph, i, _, _= inp
            # accel = self.qp_act_fn(new_graph ,graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)
            # accel = self.env.clip_action(accel)
            # graph = new_graph
            # new_graph,_, _, _, _ = self.env.step(graph, accel)
            # new_state = new_graph.env_states.agent
            # i = i + 1
            # return (new_graph, graph, i, new_state, accel)
            
        (new_graph, graph), (new_state, accel) = jax.lax.scan(
            body_fn,
            (new_graph, graph),
            None,
            length=iter_max,
        )
            
        # graph,_, _, state, accel = jax.lax.while_loop(
        #     lambda input: input[2] < iter_max,
        #     body_fn,
        #     (new_graph, graph, 0, graph.env_states.agent, jnp.zeros((graph.env_states.agent.shape[0], 2))),
        # )
        
        return new_graph, new_state, accel[-1]