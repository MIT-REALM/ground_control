import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle

from typing import Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jax import lax
from equinox.debug import breakpoint_if

from .module.root_finder import RootFinder
from ..utils.typing import Action, Params, PRNGKey, Array, List, FloatScalar
from ..utils.graph import GraphsTuple
from ..utils.utils import merge01, jax_vmap, tree_merge, tree_index, tree_where
from ..trainer.data import Rollout
from ..trainer.buffer import ReplayBuffer
from ..trainer.utils import rollout as rollout_fn
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..env.base import MultiAgentEnv
from ..algo.module.value import CostValueNet, ConstraintValueNet
from ..algo.module.policy import PPOPolicy
from ..algo.module.ef_wrapper import EFWrapper, ZEncoder
from .utils import compute_gae, compute_efocp_gae, compute_efocp_V, compute_dec_efocp_gae, compute_dec_efocp_V
from .base import Algorithm


class EFMARL(Algorithm):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            cost_weight: float = 0.,
            actor_gnn_layers: int = 2,
            critic_gnn_layers: int = 2,
            Vh_gnn_layers: int = 1,
            gamma: float = 0.99,
            lr_actor: float = 1e-5,
            lr_critic: float = 1e-5,
            batch_size: int = 8192,  # 4096,
            epoch_ppo: int = 1,
            clip_eps: float = 0.25,
            gae_lambda: float = 0.95,
            coef_ent: float = 1e-2,
            max_grad_norm: float = 2.0,
            seed: int = 0,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            rnn_step: int = 16,
            rollout_length: Optional[int] = None,
            **kwargs
    ):
        super(EFMARL, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.cost_weight = cost_weight
        self.actor_gnn_layers = actor_gnn_layers
        self.critic_gnn_layers = critic_gnn_layers
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.rollout_length = rollout_length
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.rnn_step = rnn_step
        self.z_min = -env.reward_max
        self.z_max = -env.reward_min

        # set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph
        # self.nominal_z = jnp.zeros((1,))
        self.nominal_z = jnp.array([[0.5]]).repeat(self.n_agents, axis=0)  # (n_agents, 1)

        # set up EFPPO policy
        self.policy = PPOPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.actor_gnn_layers,
            gnn_out_dim=64,
            use_lstm=False,
            use_ef=True
        )

        # initialize the rnn state
        key = jr.PRNGKey(seed)
        rnn_state_key, key = jr.split(key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_rnn_state = jax_vmap(self.policy.initialize_carry)(rnn_state_key)  # (n_agents, rnn_state_dim)
        if type(init_rnn_state) is tuple:
            init_rnn_state = jnp.stack(init_rnn_state, axis=1)  # (n_agents, n_carries, rnn_state_dim)
        else:
            init_rnn_state = jnp.expand_dims(init_rnn_state, axis=1)
        self.init_rnn_state = init_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        # initialize the policy
        policy_key, key = jr.split(key)
        policy_params = self.policy.dist.init(
            policy_key, nominal_graph, self.init_rnn_state, self.n_agents, self.nominal_z
        )
        policy_optim = optax.adam(learning_rate=lr_actor)
        self.policy_optim = optax.apply_if_finite(policy_optim, 1_000_000)
        self.policy_train_state = TrainState.create(
            apply_fn=self.policy.sample_action,
            params=policy_params,
            tx=self.policy_optim
        )

        # set up PPO critic
        self.critic = CostValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            n_out=1,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.critic_gnn_layers,
            gnn_out_dim=64,
            use_lstm=False,
            use_ef=True,
            decompose=False,
        )

        # initialize the rnn state
        rnn_state_key, key = jr.split(key)
        init_Vl_rnn_state = self.critic.initialize_carry(rnn_state_key)  # (rnn_state_dim,)
        if type(init_Vl_rnn_state) is tuple:
            init_Vl_rnn_state = jnp.stack(init_Vl_rnn_state, axis=0)  # (n_carries, rnn_state_dim)
        else:
            init_Vl_rnn_state = init_Vl_rnn_state[None, :]
        # (n_rnn_layers, 1, n_carries, rnn_state_dim)
        self.init_Vl_rnn_state = init_Vl_rnn_state[None, :, :].repeat(self.rnn_layers, axis=0)[:, None, :, :]

        # initialize the critic
        critic_key, key = jr.split(key)
        critic_params = self.critic.net.init(
            critic_key, nominal_graph, self.init_Vl_rnn_state, self.n_agents, self.nominal_z[0][None, :])
        critic_optim = optax.adam(learning_rate=lr_critic)
        self.critic_optim = optax.apply_if_finite(critic_optim, 1_000_000)
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.get_value,
            params=critic_params,
            tx=self.critic_optim
        )

        # set up constraint value net
        self.Vh = ConstraintValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            n_out=env.n_cost,
            gnn_layers=Vh_gnn_layers,
            gnn_out_dim=64,
            use_ef=True,
            decompose=True
        )

        Vh_key, key = jr.split(key)
        Vh_params = self.Vh.net.init(Vh_key, nominal_graph, self.n_agents, self.nominal_z)
        Vh_optim = optax.adam(learning_rate=lr_critic)
        self.Vh_optim = optax.apply_if_finite(Vh_optim, 1_000_000)
        self.Vh_train_state = TrainState.create(
            apply_fn=self.Vh.get_value,
            params=Vh_params,
            tx=self.Vh_optim
        )

        # set up the root finder
        self.root_finder = RootFinder(
            z_min=self.z_min,
            z_max=self.z_max,
            n_agent=self.n_agents
        )

        # set up key
        self.key = key

        # rollout function
        def rollout_fn_single_(cur_params, cur_key, cur_init_graph):
            return rollout_fn(self._env,
                              ft.partial(self.step, params=cur_params),
                              self.init_rnn_state,
                              cur_key,
                              self.gamma,
                              cur_init_graph)

        def rollout_fn_(cur_params, cur_keys, cur_init_graphs=None):
            return jax.vmap(ft.partial(rollout_fn_single_, cur_params))(cur_keys, cur_init_graphs)

        self.rollout_fn = jax.jit(rollout_fn_)

    @property
    def config(self) -> dict:
        return {
            'cost_weight': self.cost_weight,
            'actor_gnn_layers': self.actor_gnn_layers,
            'critic_gnn_layers': self.critic_gnn_layers,
            'gamma': self.gamma,
            'rollout_length': self.rollout_length,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'batch_size': self.batch_size,
            'epoch_ppo': self.epoch_ppo,
            'clip_eps': self.clip_eps,
            'gae_lambda': self.gae_lambda,
            'coef_ent': self.coef_ent,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'rnn_step': self.rnn_step,
            'z_min': self.z_min,
            'z_max': self.z_max
        }

    @property
    def params(self) -> Params:
        return {
            "policy": self.policy_train_state.params,
            "Vl": self.critic_train_state.params,
            "Vh": self.Vh_train_state.params
        }

    def get_opt_z(self, graph: GraphsTuple, params: Optional[Params] = None) -> Tuple[FloatScalar, Array]:
        if params is None:
            params = self.params

        def fn_(Vh_params, obs):
            Vh_fn = ft.partial(self.Vh_train_state.apply_fn, Vh_params, obs)
            return self.root_finder.get_dec_opt_z(Vh_fn, obs)

        return jax.jit(fn_)(params["Vh"], graph)

    def act(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, params: Optional[Params] = None
    ) -> [Action, Array]:
        if params is None:
            params = self.params["policy"]
        action, rnn_state = self.policy.get_action(params, graph, rnn_state, z)
        return action, rnn_state

    def get_value(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, params: Optional[Params] = None
    ) -> Tuple[Array, Array]:
        if params is None:
            params = self.params
        value, rnn_state = self.critic.get_value(params["Vl"], graph, rnn_state, z[0][None, :])
        return value, rnn_state

    def get_Vh(
            self, graph: GraphsTuple, z: Array, params: Optional[Params] = None
    ) -> Array:
        if params is None:
            params = self.params["Vh"]
        Vh = self.Vh.get_value(params, graph, z)
        return Vh

    def step(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, key: PRNGKey, params: Optional[Params] = None
    ) -> Tuple[Action, Array, Array]:
        if params is None:
            params = self.params
        action, log_pi, rnn_state = self.policy_train_state.apply_fn(params["policy"], graph, rnn_state, key, z)
        return action, log_pi, rnn_state

    def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
        rollout_result = self.rollout_fn(params, b_key)
        return rollout_result

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        update_info = {}
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        for i_epoch in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, rollout.dones.shape[1] // self.rnn_step))
            batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // (self.batch_size // rollout.dones.shape[1])))
            critic_train_state, Vh_train_state, policy_train_state, update_info = self.update_inner(
                self.critic_train_state,
                self.Vh_train_state,
                self.policy_train_state,
                rollout,
                batch_idx,
                rnn_chunk_ids,
            )
            self.critic_train_state = critic_train_state
            self.policy_train_state = policy_train_state
            self.Vh_train_state = Vh_train_state
        return update_info

    def scan_value(
            self,
            rollout: Rollout,
            init_rnn_state_Vl: Array,
            Vl_params: Params,
    ) -> Tuple[Array, Array, Array]:
        graphs = rollout.graph  # (T,)
        zs = rollout.zs  # (T, a, 1)

        def body_(carry, inp):
            graph, z = inp
            rnn_state = carry
            value, new_rnn_state = self.critic.get_value(Vl_params, graph, rnn_state, z[0][None, :])
            return new_rnn_state, (value, rnn_state)

        final_rnn_state_Vl, (T_Vl, rnn_states_Vl) = (
            jax.lax.scan(body_, init_rnn_state_Vl, (graphs, zs)))

        T_Vl = T_Vl.squeeze()
        return T_Vl, rnn_states_Vl, final_rnn_state_Vl

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            critic_train_state: TrainState,
            Vh_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
    ) -> Tuple[TrainState, TrainState, TrainState, dict]:
        # rollout: (n_env, T, n_agent, ...)
        b, T, a, _ = rollout.zs.shape

        # get final z
        b_final_z = (rollout.zs[:, -1] + rollout.rewards[:, -1, None, None]) / self.gamma  # (b, a, 1)

        # calculate Vl
        scan_value = ft.partial(self.scan_value,
                                init_rnn_state_Vl=self.init_Vl_rnn_state,
                                Vl_params=critic_train_state.params)
        bT_Vl, rnn_states_Vl, final_rnn_states_Vl = jax.vmap(scan_value)(rollout)

        def final_value_fn_(graph, final_z, rnn_state_Vl):
            value, _ = self.critic.get_value(
                critic_train_state.params, tree_index(graph, -1), rnn_state_Vl, final_z[0][None, :])
            return value.squeeze()

        final_Vl = jax.vmap(final_value_fn_)(rollout.next_graph, b_final_z, final_rnn_states_Vl)
        bTp1_Vl = jnp.concatenate([bT_Vl, final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape == (b, T + 1)

        # calculate Vh
        Vh_fn = ft.partial(self.get_Vh, params=Vh_train_state.params)
        bTah_Vh = jax.vmap(jax.vmap(Vh_fn))(rollout.graph, rollout.zs)
        final_Vh = jax.vmap(Vh_fn)(jtu.tree_map(lambda x: x[:, -1], rollout.next_graph), b_final_z)

        # def final_Vh_fn_(graph, zs, rnn_state):
        #     _, final_rnn_state = self.act(tree_index(graph, -1), zs[-1], rnn_state[-1], policy_train_state.params)
        #     return self.get_Vh(tree_index(graph, -1), zs[-1], final_rnn_state, Vh_train_state.params)

        # final_Vh = jax.vmap(final_Vh_fn_)(rollout.next_graph, rollout.zs, rollout.rnn_states)
        bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)
        assert bTp1ah_Vh.shape == (b, T + 1, a, self._env.n_cost)

        # calculate Dec-EFOCP GAE
        bTah_Qh, bT_Ql, bTa_Q = jax.vmap(
            ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=rollout.costs,
          T_l=-rollout.rewards,
          T_z=rollout.zs.squeeze(-1)[:, :, 0],
          Tp1ah_Vh=bTp1ah_Vh,
          Tp1_Vl=bTp1_Vl)

        # calculate advantages and normalize
        bTa_V = jax_vmap(jax_vmap(compute_dec_efocp_V))(rollout.zs.squeeze(-1)[:, :, 0], bTah_Vh, bT_Vl)
        assert bTa_V.shape == (b, T, a)
        bTa_A = bTa_Q - bTa_V
        bTa_A = (bTa_A - bTa_A.mean(axis=1, keepdims=True)) / (bTa_A.std(axis=1, keepdims=True) + 1e-8)
        assert bTa_A.shape == (b, T, a)

        # update ppo
        def update_fn(carry, idx):
            critic, Vh, policy = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            critic, Vh, value_info = self.update_value(
                critic, Vh, rollout_batch, bT_Ql[idx], bTah_Qh[idx],
                rnn_states_Vl[idx], rollout.rnn_states[idx], rnn_chunk_ids
            )
            policy, policy_info = self.update_policy(policy, rollout_batch, bTa_A[idx], rnn_chunk_ids)
            return (critic, Vh, policy), (value_info | policy_info)

        (critic_train_state, Vh_train_state, policy_train_state), info = lax.scan(
            update_fn, (critic_train_state, Vh_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)

        return critic_train_state, Vh_train_state, policy_train_state, info

    def scan_eval_action(
            self, rollout: Rollout, init_rnn_state: Array, action_keys: PRNGKey, actor_params: Params
    ) -> Tuple[Array, Array, Array, Array]:
        T_graph = rollout.graph  # (T, )
        Ta_z = rollout.zs  # (T, n_agent, 1)
        Ta_action = rollout.actions  # (T, n_agents, action_dim)

        def body_(rnn_state, inp):
            graph, z, key, action = inp
            log_pi, entropy, new_rnn_state = self.policy.eval_action(actor_params, graph, action, rnn_state, key, z)
            return new_rnn_state, (log_pi, entropy, rnn_state)

        final_rnn_state, outputs = jax.lax.scan(body_, init_rnn_state, (T_graph, Ta_z, action_keys, Ta_action))
        Ta_log_pis, Ta_entropies, rnn_states = outputs

        return Ta_log_pis, Ta_entropies, rnn_states, final_rnn_state

    def update_policy(
            self,
            policy_train_state: TrainState,
            rollout: Rollout,
            bTa_A: Array,
            rnn_chunk_ids: Array
    ):
        # divide the rollout into chunks (n_env, n_chunks, T, ...)
        bcT_rollout = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout)
        rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])
        bcTa_A = bTa_A[:, rnn_chunk_ids]

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        action_keys = jax.tree_map(lambda x: x[:, rnn_chunk_ids], action_keys)

        def get_loss(params):
            bcTa_log_pis, bcTa_entropy, _, _ = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action, actor_params=params)
            ))(bcT_rollout, rnn_state_inits, action_keys)
            bcTa_ratio = jnp.exp(bcTa_log_pis - bcT_rollout.log_pis)
            loss_policy1 = bcTa_ratio * bcTa_A
            loss_policy2 = jnp.clip(bcTa_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * bcTa_A
            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            mean_entropy = bcTa_entropy.mean()
            policy_loss = loss_policy - self.coef_ent * mean_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(bcTa_ratio - 1.0))
            info = {
                'policy/loss': loss_policy,
                'policy/entropy': mean_entropy,
                'policy/clip_frac': clip_frac,
                'policy/total_variation_dist': total_variation_dist
            }
            return policy_loss, info

        grad, policy_info = jax.grad(get_loss, has_aux=True)(policy_train_state.params)
        grad_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        return policy_train_state, (policy_info | {'policy/has_nan': grad_has_nan, 'policy/grad_norm': grad_norm})

    def update_value(
            self,
            critic_train_state: TrainState,
            Vh_train_state: TrainState,
            rollout: Rollout,
            bT_Ql: Array,
            bTah_Qh: Array,
            rnn_states_Vl: Array,
            rnn_states_Vh: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        bcT_rollout = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout)  # (n_env, n_chunk, T, ...)
        Vl_rnn_state_inits = jnp.zeros_like(rnn_states_Vl[:, rnn_chunk_ids[:, 0]])  # (n_env, n_chunk, ...)
        # Vh_rnn_state_inits = jnp.zeros_like(rnn_states_Vh[:, rnn_chunk_ids[:, 0]])
        # bcT_rnn_states_Vh = rnn_states_Vh[:, rnn_chunk_ids]
        bcT_Ql = bT_Ql[:, rnn_chunk_ids]
        bcTah_Qh = bTah_Qh[:, rnn_chunk_ids]

        def get_loss(critic_params, Vh_params):
            bcT_Vl, _, _ = jax.vmap(jax.vmap(
                ft.partial(self.scan_value, Vl_params=critic_params)))(bcT_rollout, Vl_rnn_state_inits)
            loss_Vl = optax.l2_loss(bcT_Vl, bcT_Ql).mean()
            bcTah_Vh = jax.vmap(jax.vmap(jax.vmap(
                ft.partial(self.get_Vh, params=Vh_params))))(bcT_rollout.graph, bcT_rollout.zs)
            loss_Vh = optax.l2_loss(bcTah_Vh, bcTah_Qh).mean()
            info = {
                'critic/loss': loss_Vl,
                'critic/loss_Vh': loss_Vh,
                'critic/gt_unsafe': (bcTah_Qh > 0).mean()
            }
            return loss_Vl + loss_Vh, info

        (grad_Vl, grad_Vh), value_info = jax.grad(get_loss, argnums=(0, 1), has_aux=True)(
            critic_train_state.params, Vh_train_state.params)
        grad_Vl_has_nan = has_any_nan_or_inf(grad_Vl).astype(jnp.float32)
        grad_Vh_has_nan = has_any_nan_or_inf(grad_Vh).astype(jnp.float32)
        grad_Vl, grad_Vl_norm = compute_norm_and_clip(grad_Vl, self.max_grad_norm)
        grad_Vh, grad_Vh_norm = compute_norm_and_clip(grad_Vh, self.max_grad_norm)
        critic_train_state = critic_train_state.apply_gradients(grads=grad_Vl)
        Vh_train_state = Vh_train_state.apply_gradients(grads=grad_Vh)

        return critic_train_state, Vh_train_state, (value_info | {'critic/has_nan': grad_Vl_has_nan,
                                                                  'critic/grad_Vh_has_nan': grad_Vh_has_nan,
                                                                  'critic/grad_norm': grad_Vl_norm,
                                                                  'critic/grad_Vh_norm': grad_Vh_norm})

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.critic_train_state.params, open(os.path.join(model_dir, 'critic.pkl'), 'wb'))
        pickle.dump(self.Vh_train_state.params, open(os.path.join(model_dir, 'Vh.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.critic_train_state = \
            self.critic_train_state.replace(params=pickle.load(open(os.path.join(path, 'critic.pkl'), 'rb')))
        self.Vh_train_state = \
            self.Vh_train_state.replace(params=pickle.load(open(os.path.join(path, 'Vh.pkl'), 'rb')))
