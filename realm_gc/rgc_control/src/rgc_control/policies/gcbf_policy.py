import functools as ft
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml


from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.policy import ControlPolicy

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env



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
        step=10
        
        env = make_env(
            env_id='F1TenthNew' if step == 500 else 'DubinsCarAdapt',
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
        print('name: ', env.env_name())
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
        self.state_dim = env.state_dim
        
        _ = self.qp_act_fn(graph0, graph0)
        self.env = env
        self.step = jax.jit(env.step)
        self.get_states_fn = jax.jit(self.get_states)
        _ = self.get_states_fn(graph0, graph0, None, None, 2)
        _ = self.step(graph0, jnp.zeros((1,2)))
        self.graph0 = graph0
        self.car_goal = car_goal
        self.graph0 = self.create_graph(car_pos, car_goal, obs_pos)
        self.obs = obs_pos
        
    def create_graph(self, car_pos, car_goal, obs_pos, graph=None):
        if graph is None:
            graph = self.graph0
        states=graph.env_states
        obs = states.obstacle
        if self.env.env_name() == 'Dubins':
            agent_states= car_pos[None, :]
        else:
            agent_states = jnp.array([car_pos[0], car_pos[1], jnp.cos(car_pos[2]), jnp.sin(car_pos[2]), car_pos[3]])[None, :]
        
        if self.env.env_name() == 'Dubins':
            goal_states = car_goal[None, :]
        else:
            goal_states = jnp.array([car_goal[0], car_goal[1], jnp.cos(car_goal[2]), jnp.sin(car_goal[2]), car_goal[3]])[None, :]
        

        # mov_obs = jnp.concatenate([obs_pos[:,0], obs_pos[:,1], jnp.zeros(obs_pos.shape[0]), jnp.zeros(obs_pos.shape[0])], axis=0)
        mov_obs = obs_pos
        if self.env.env_name() == 'Dubins':
            mov_obs = jnp.concatenate([mov_obs, jnp.zeros((mov_obs.shape[0], 2))], axis=1)
        else:
            mov_obs = jnp.concatenate([mov_obs, jnp.zeros((mov_obs.shape[0], 3))], axis=1)

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
            ref_vel = jnp.array([ref_inp.steering_angle, ref_inp.acceleration])
        else:
            ref_vel = None

        mov_obs_vel = self.env.mov_obs_vel_pred(new_graph)

        ref_accel, flag = self.ref_check_fn(new_graph, graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)
        # flag = 0

        if flag == 1:
            accel = ref_accel[None, :]
            next_graph, _, _, _, _  = self.env.step(new_graph, accel)
            next_state = next_graph.env_states.agent
            if self.env.env_name() == 'F1Tenth':
                next_state = jnp.array([next_state[0], next_state[1], jnp.arctan2(next_state[2], next_state[3]), next_state[4]])
            return F1TenthAction(
                acceleration=accel[0, 1],
                steering_angle=accel[0, 0],
            ), next_state.squeeze(), flag
        else:
            num_iter = 50
            states, accel = self.get_states_fn(new_graph, graph, mov_obs_vel, ref_vel, num_iter)
            # states = jnp.zeros((num_iter, graph.env_states.agent.shape[1]))
            # for j in range(num_iter):
            #     state = graph.env_states.agent
            #     print('prev state: ', state)
                
            #     accel = self.qp_act_fn(new_graph, graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)
            #     # accel = self.act_fn(new_graph)
            #     accel = self.env.clip_action(accel)
                
            #     new_graph, _, _, _, _ = self.step(new_graph, accel)
                
            #     obs_coll = self.env.mov_obs_collision_mask(new_graph)

            #     print('obs coll:', obs_coll * 1)
                
            #     next_state = new_graph.env_states.agent
            #     # print('accel: ', accel)
            #     print('graph state after step: ', new_graph.env_states.agent)
            #     states = states.at[j].set(next_state.squeeze())
        
            next_state = states
            if self.env.env_name() == 'F1Tenth':
                next_state = jnp.array([next_state[:, 0], next_state[:, 1], jnp.arctan2(next_state[:, 2], next_state[:, 3]), next_state[:, 4]])
        return F1TenthAction(
            acceleration=accel[0, 1],
            steering_angle=accel[0, 0],
        ), next_state.squeeze(), flag
    
    def get_states(self, new_graph, graph, mov_obs_vel, ref_vel, num_iter):
        states = jnp.zeros((20, self.state_dim))
        def body(init):
            new_graph = init[0]
            graph = init[1]
            j = init[2]
            states = init[3]
            accel = self.qp_act_fn(new_graph, graph, mov_obs_vel=mov_obs_vel, ref_in=ref_vel)
            # accel = self.act_fn(new_graph)
            graph = new_graph
            accel = self.env.clip_action(accel)
            
            new_graph, _, _, _, _ = self.step(new_graph, accel)
            next_state = new_graph.env_states.agent
            states = states.at[j].set(next_state.squeeze())
            j = j+1

            return [new_graph, graph, j, states, accel]
        
        def cond_fn(init):
            return init[2]< num_iter

        _, _, _, new_states, accel = jax.lax.while_loop(cond_fn, body, [new_graph, graph, 0, states, jnp.zeros((1, 2))])

        return new_states, accel
        


 