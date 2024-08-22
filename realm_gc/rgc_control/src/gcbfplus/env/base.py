import functools as ft
import numpy as np
import pathlib
import jax
import jax.lax as lax
import jax.numpy as jnp
import tqdm

from scipy.interpolate import CubicSpline

from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, NamedTuple, Optional, Tuple

from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, PRNGKey, Reward, State
from ..utils.utils import jax2np, jax_jit_np, tree_concat_at_front, tree_stack


class StepResult(NamedTuple):
    graph: GraphsTuple
    reward: Reward
    cost: Cost
    done: Done
    info: Info


class RolloutResult(NamedTuple):
    Tp1_graph: GraphsTuple
    T_action: Action
    T_reward: Reward
    T_cost: Cost
    T_done: Done
    T_info: Info


class MultiAgentEnv(ABC):

    PARAMS = {}

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        super(MultiAgentEnv, self).__init__()
        self._num_agents = num_agents
        self._dt = dt
        if params is None:
            params = self.PARAMS
        self._params = params
        self._t = 0
        self._max_step = max_step
        self._max_travel = max_travel
        self._area_size = area_size
        self._mov_obs = None
        self._agent_states = None
        self.vel_pred_fn = jax.jit(jax.vmap(self.vel_pred))

    @property
    def params(self) -> dict:
        return self._params

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def max_travel(self) -> float:
        return self._max_travel

    @property
    def area_size(self) -> float:
        return self._area_size

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def max_episode_steps(self) -> int:
        return self._max_step

    def clip_state(self, state: State) -> State:
        lower_limit, upper_limit = self.state_lim(state)
        return jnp.clip(state, lower_limit, upper_limit)

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    @abstractproperty
    def state_dim(self) -> int:
        pass

    @abstractproperty
    def node_dim(self) -> int:
        pass

    @abstractproperty
    def edge_dim(self) -> int:
        pass

    @abstractproperty
    def action_dim(self) -> int:
        pass

    @abstractmethod
    def mov_obsvel(self) -> Array:
        pass
    
    @abstractmethod
    def reset(self, key: Array) -> GraphsTuple:
        pass

    def reset_np(self, key: Array) -> GraphsTuple:
        """Reset, but without the constraint that it has to be jittable."""
        return self.reset(key)

    @abstractmethod
    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False) -> StepResult:
        pass

    @abstractmethod
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        pass

    @abstractmethod
    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        pass

    @abstractmethod
    def control_affine_dyn(self, state: State) -> [Array, Array]:
        pass

    @abstractmethod
    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        pass

    @abstractmethod
    def get_graph(self, state: State) -> GraphsTuple:
        pass

    @abstractmethod
    def u_ref(self, graph: GraphsTuple) -> Action:
        pass

    @abstractmethod
    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        pass

    @abstractmethod
    def collision_mask(self, graph: GraphsTuple) -> Array:
        pass
    
    def vel_pred(self, all_states):
        '''
        input: array of states
        output: predicted next velocity based on spline fitting
        '''
        ind = jnp.arange(all_states.shape[0]).astype(jnp.float32) * self.dt
        ext_ind = jnp.arange(all_states.shape[0], all_states.shape[0]+1).astype(jnp.float32) * self.dt
        
        # polyx = CubicSpline(ind, all_states[:,0], bc_type='natural')
        # polyy = CubicSpline(ind, all_states[:,1], bc_type='natural')
        polyx = jnp.polyfit(ind, all_states[:,0], 3)
        polyy = jnp.polyfit(ind, all_states[:,1], 3)
        # new_x = polyx(ext_ind)
        # new_y = polyy(ext_ind)
        new_x = jnp.polyval(polyx, ext_ind)
        new_y = jnp.polyval(polyy, ext_ind)
        velx = (new_x - all_states[-1,0]) / self.dt
        vely = (new_y - all_states[-1,1]) / self.dt
        # spline_fitx = UnivariateSpline(ind, all_states[:,0], k=3)
        # spline_fity = UnivariateSpline(ind, all_states[:,1], k=3)
        # spline_fitx.set_smoothing_factor(0.1)
        # spline_fity.set_smoothing_factor(0.1)
        # velx = spline_fitx.derivative(n=1)(ind)
        # vely = spline_fity.derivative(n=1)(ind)
        return jnp.array([velx, vely]).reshape(1,2)
    
    def mov_agent_vel_pred(self,graph: GraphsTuple = None, state:Array = None) -> Array:
        if graph is not None:
            mov_agent_new = graph.type_states(type_idx=0, n_type=self.num_agents)
        else:
            mov_agent_new = state
        # vel_pred_fn = jax.jit(jax.vmap(self.vel_pred))
        if self._agent_states is None:
            self._agent_states = mov_agent_new[:,None,:].repeat(5, axis=1)
        else:
            self._agent_states = jnp.concatenate([self._agent_states, mov_agent_new[:, None, :]], axis=1)
        # breakpoint()
            self._agent_states = self._agent_states[:, -5:, :]
        
        vel = self.vel_pred_fn(self._agent_states)
        

        return vel.squeeze()
    
    def mov_obs_vel_pred(self,graph: GraphsTuple = None, state: Array = None) -> Array:
        if graph is not None:
            mov_obs_new = graph.env_states.mov_obs
        else:
            mov_obs_new = state
        
        if self._mov_obs is None:
            self._mov_obs = mov_obs_new[:,None,:].repeat(5, axis=1)
        else:
            self._mov_obs = jnp.concatenate([self._mov_obs, mov_obs_new[:, None, :]], axis=1)
        # breakpoint()
            self._mov_obs = self._mov_obs[:, -5:, :]
        
        vel = self.vel_pred_fn(self._mov_obs)
        vel = jnp.concatenate([vel, 0*vel], axis=-1)

        return vel.squeeze()
    
    def rollout_fn(self, policy: Callable, rollout_length: int = None) -> Callable[[PRNGKey], RolloutResult]:
        rollout_length = rollout_length or self.max_episode_steps

        def body(graph, _):
            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return graph_new, (graph_new, action, reward, cost, done, info)

        def fn(key: PRNGKey) -> RolloutResult:
            graph0 = self.reset(key)
            graph_final, (T_graph, T_action, T_reward, T_cost, T_done, T_info) = lax.scan(
                body, graph0, None, length=rollout_length
            )
            Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)

            return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        return fn

    def rollout_qp_fn(self, policy: Callable, rollout_length: int = None) -> Callable[[PRNGKey], RolloutResult]:
        rollout_length = rollout_length or self.max_episode_steps

        def body(init, _):
            graph = init[0]
            prev_graph = init[1]
            vel = self.mov_obs_vel_pred(graph)
            action = policy(graph, prev_graph,mov_obs_vel=vel)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return [graph_new, graph], (graph_new, action, reward, cost, done, info)

        def fn(key: PRNGKey) -> RolloutResult:
            graph0 = self.reset(key)
            [graph_final, graph], (T_graph, T_action, T_reward, T_cost, T_done, T_info) = lax.scan(
                body, [graph0, graph0], None, length=rollout_length
            )
            Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)

            return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        return fn
    
    def rollout_fn_jitstep(
        self, policy: Callable, rollout_length: int = None, noedge: bool = False, nograph: bool = False
    ):
        rollout_length = rollout_length or self.max_episode_steps

        def body(graph, _):
            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return graph_new, (graph_new, action, reward, cost, done, info)

        jit_body = jax.jit(body)

        is_unsafe_fn = jax_jit_np(self.collision_mask)
        is_finish_fn = jax_jit_np(self.finish_mask)

        def fn(key: PRNGKey) -> [RolloutResult, Array, Array]:
            graph0 = self.reset_np(key)
            graph = graph0
            T_output = []
            is_unsafes = []
            is_finishes = []

            is_unsafes.append(is_unsafe_fn(graph0))
            is_finishes.append(is_finish_fn(graph0))
            graph0 = jax2np(graph0)

            for kk in tqdm.trange(rollout_length, ncols=80):
                graph, output = jit_body(graph, None)

                is_unsafes.append(is_unsafe_fn(graph))
                is_finishes.append(is_finish_fn(graph))

                output = jax2np(output)
                if noedge:
                    output = (output[0].without_edge(), *output[1:])
                if nograph:
                    output = (None, *output[1:])
                T_output.append(output)

            # Concatenate everything together.
            T_graph = [o[0] for o in T_output]
            if noedge:
                T_graph = [graph0.without_edge()] + T_graph
            else:
                T_graph = [graph0] + T_graph
            del graph0
            T_action = [o[1] for o in T_output]
            T_reward = [o[2] for o in T_output]
            T_cost = [o[3] for o in T_output]
            T_done = [o[4] for o in T_output]
            T_info = [o[5] for o in T_output]
            del T_output

            if nograph:
                T_graph = None
            else:
                T_graph = tree_stack(T_graph)
            T_action = tree_stack(T_action)
            T_reward = tree_stack(T_reward)
            T_cost = tree_stack(T_cost)
            T_done = tree_stack(T_done)
            T_info = tree_stack(T_info)

            Tp1_graph = T_graph

            rollout_result = jax2np(RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info))
            return rollout_result, np.stack(is_unsafes, axis=0), np.stack(is_finishes, axis=0)

        return fn

    @abstractmethod
    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def finish_mask(self, graph: GraphsTuple) -> Array:
        pass
