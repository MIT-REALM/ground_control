import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax
from equinox.debug import breakpoint_if

from .module.root_finder import RootFinder
from ..utils.typing import Action, Params, PRNGKey, Array, List, FloatScalar
from ..utils.graph import GraphsTuple
from ..utils.utils import merge01, jax_vmap, tree_merge, tree_index, tree_where
from ..trainer.data import Rollout
from ..trainer.buffer import ReplayBuffer
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..trainer.utils import rollout as rollout_fn
from ..env.base import MultiAgentEnv
from ..algo.module.value import CostValueNet
from ..algo.module.policy import PPOPolicy
from ..algo.module.ef_wrapper import EFWrapper, ZEncoder
from .utils import compute_gae, compute_efocp_gae, compute_efocp_V, compute_dec_efocp_gae, compute_dec_efocp_V
from .base import Algorithm


class InformarlCRPO(Algorithm):

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
            use_lstm: bool = False,
            **kwargs
    ):
        super(InformarlCRPO, self).__init__(
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
        self.use_lstm = use_lstm

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
            use_lstm=self.use_lstm,
            use_ef=False
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
        self.nominal_z = jnp.array([[0.5]]).repeat(self.n_agents, axis=0)  # (n_agents, 1)

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
            use_lstm=self.use_lstm,
            use_ef=False,
            decompose=True
        )

        # initialize the rnn state
        rnn_state_key, key = jr.split(key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_value_rnn_state = jax_vmap(self.critic.initialize_carry)(rnn_state_key)  # (n_agents, rnn_state_dim)
        if type(init_value_rnn_state) is tuple:
            init_value_rnn_state = jnp.stack(init_value_rnn_state, axis=1)  # (n_agents, n_carries, rnn_state_dim)
        else:
            init_value_rnn_state = jnp.expand_dims(init_value_rnn_state, axis=1)
        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_value_rnn_state = init_value_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        # initialize the critic
        critic_key, key = jr.split(key)
        critic_params = self.critic.net.init(
            critic_key, nominal_graph, self.init_value_rnn_state, self.n_agents, self.nominal_z)
        critic_optim = optax.adam(learning_rate=lr_critic)
        self.critic_optim = optax.apply_if_finite(critic_optim, 1_000_000)
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.get_value,
            params=critic_params,
            tx=self.critic_optim
        )

        # set up constraint value net
        self.Vh = CostValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            n_out=env.n_cost,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=1,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            use_ef=False,
            decompose=True
        )
        Vh_key, key = jr.split(key)
        Vh_params = self.Vh.net.init(Vh_key, nominal_graph, self.init_value_rnn_state, self.n_agents, self.nominal_z)
        Vh_optim = optax.adam(learning_rate=lr_critic)
        self.Vh_optim = optax.apply_if_finite(Vh_optim, 1_000_000)
        self.Vh_train_state = TrainState.create(
            apply_fn=self.Vh.get_value,
            params=Vh_params,
            tx=self.Vh_optim
        )

        # set up key
        self.key = key

        # rollout function
        def rollout_fn_single_(cur_params, cur_key):
            return rollout_fn(self._env,
                              ft.partial(self.step, params=cur_params),
                              ft.partial(self.get_value, params=cur_params),
                              self.init_rnn_state,
                              self.init_value_rnn_state,
                              cur_key,
                              self.gamma)

        def rollout_fn_(cur_params, cur_keys):
            return jax.vmap(ft.partial(rollout_fn_single_, cur_params))(cur_keys)

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
            'use_lstm': self.use_lstm,
        }

    @property
    def params(self) -> Params:
        return {
            "policy": self.policy_train_state.params,
            "Vl": self.critic_train_state.params,
            "Vh": self.Vh_train_state.params
        }

    def act(
            self,
            graph: GraphsTuple,
            z: Array,
            rnn_state: Array,
            params: Optional[Params] = None,
    ) -> [Action, Array]:
        if params is None:
            params = self.params
        action, rnn_state = self.policy.get_action(params["policy"], graph, rnn_state, z)
        return action, rnn_state

    def get_value(
            self,
            graph: GraphsTuple,
            z: Array,
            rnn_state: Array,
            params: Optional[Params] = None
    ) -> Tuple[Array, Array]:
        if params is None:
            params = self.params
        value, rnn_state = self.critic.get_value(params["Vl"], graph, rnn_state, z)
        return value, rnn_state

    def step(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, key: PRNGKey, params: Optional[Params] = None
    ) -> Tuple[Action, Array, Array]:
        if params is None:
            params = self.params
        action, log_pi, rnn_state = self.policy_train_state.apply_fn(params["policy"], graph, rnn_state, key, z)
        return action, log_pi, rnn_state

    def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
        # init_rollout_key = jax.vmap(jr.split)(b_key)
        # init_key = init_rollout_key[:, 0]
        # rollout_key = init_rollout_key[:, 1]
        # init_graphs = jax.vmap(ft.partial(self.get_init_graph, memory=self.memory))(init_key)

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
                self.critic_train_state, self.Vh_train_state, self.policy_train_state, rollout, batch_idx, rnn_chunk_ids
            )
            self.critic_train_state = critic_train_state
            self.policy_train_state = policy_train_state
            self.Vh_train_state = Vh_train_state
        # self.memory = rollout
        return update_info

    def scan_value(
            self,
            rollout: Rollout,
            init_rnn_state_V: Array,
            init_rnn_state_Vh: Array,
            critic_params: Params,
            Vh_params: Params
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array], Tuple[Array, Array]]:
        graphs = rollout.graph  # (T,)
        zs = rollout.zs  # (T, a, 1)

        def body_(rnn_state, inp):
            graph, z = inp
            rnn_state_V, rnn_state_Vh = rnn_state
            value, new_rnn_state_V = self.critic.get_value(critic_params, graph, rnn_state_V, z)
            value_h, new_rnn_state_Vh = self.Vh.get_value(Vh_params, graph, rnn_state_Vh, z)
            return (new_rnn_state_V, new_rnn_state_Vh), (value, value_h, rnn_state_V, rnn_state_Vh)

        (final_rnn_state_Vl, final_rnn_state_Vh), (Ta_Vl, Tah_Vh, rnn_states_Vl, rnn_states_Vh) = (
            jax.lax.scan(body_, (init_rnn_state_V, init_rnn_state_Vh), (graphs, zs)))

        Ta_Vl = Ta_Vl.squeeze()
        # Tah_Vh = Tah_Vh.squeeze(1)
        return (Ta_Vl, Tah_Vh), (rnn_states_Vl, rnn_states_Vh), (final_rnn_state_Vl, final_rnn_state_Vh)

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            critic_train_state: TrainState,
            Vh_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, TrainState, dict]:
        # rollout: (n_env, T, n_agent, ...)

        # calculate values and next_values
        scan_value = ft.partial(self.scan_value,
                                init_rnn_state_V=self.init_value_rnn_state,
                                init_rnn_state_Vh=self.init_value_rnn_state,
                                critic_params=critic_train_state.params,
                                Vh_params=Vh_train_state.params)
        (bTa_Vl, bTah_Vh), (rnn_states_V, rnn_states_Vh), (final_rnn_states_Vl, final_rnn_states_Vh) = (
            jax_vmap(scan_value)(rollout))

        def final_value_fn(graph, zs, rnn_state_V, rnn_state_Vh):
            value, _ = self.critic.get_value(critic_train_state.params, tree_index(graph, -1), rnn_state_V, zs[-1])
            value_h, _ = self.Vh.get_value(Vh_train_state.params, tree_index(graph, -1), rnn_state_Vh, zs[-1])
            return value.squeeze(), value_h

        final_Vl, final_Vh = jax_vmap(final_value_fn)(
            rollout.next_graph, rollout.zs, final_rnn_states_Vl, final_rnn_states_Vh)
        bTp1a_Vl = jnp.concatenate([bTa_Vl, final_Vl[:, None]], axis=1)
        bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)

        bTp1_Vl = bTp1a_Vl.sum(-1)
        bTp1h_Vh = bTp1ah_Vh.reshape(bTp1ah_Vh.shape[0], bTp1ah_Vh.shape[1], -1)

        # calculate Dec-EFOCP GAE
        bTah_Qh, bT_Ql, bTa_Q = jax.vmap(
            ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=rollout.costs,
          T_l=-rollout.rewards,
          T_z=rollout.zs.squeeze(-1).sum(-1),
          Tp1ah_Vh=bTp1ah_Vh,
          Tp1_Vl=bTp1_Vl)

        # calculate advantages and normalize
        # bT_Al = bT_Ql - bTa_Vl.sum(-1)
        # bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        # bTah_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)[:, :, :, None].repeat(bTah_Vh.shape[-1], axis=-1)
        # bTah_Ah = bTah_Qh - bTah_Vh
        # bTah_Ah = (bTah_Ah - bTah_Ah.mean(axis=1, keepdims=True)) / (bTah_Ah.std(axis=1, keepdims=True) + 1e-8)
        # assert bTah_Ah.shape == bTah_Al.shape
        # bTah_is_safe = bTah_Qh <= 0
        # bTa_is_safe = bTah_is_safe.min(axis=-1)
        # bTah_A = jnp.where(bTa_is_safe[:, :, :, None], bTah_Al, jnp.zeros_like(bTah_Al))
        # bTah_A = jnp.where(bTah_is_safe, bTah_A, bTah_Ah)
        # bTa_A = bTah_A.sum(axis=-1) / (jnp.count_nonzero(bTah_A, axis=-1) + 1e-8)

        bT_Al = bT_Ql - bTa_Vl.sum(-1)
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        bTa_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)
        bTah_Ah = bTah_Qh - bTah_Vh
        bTah_Ah = (bTah_Ah - bTah_Ah.mean(axis=1, keepdims=True)) / (bTah_Ah.std(axis=1, keepdims=True) + 1e-8)
        # assert bTah_Ah.shape == bTah_Al.shape
        bTah_is_safe = bTah_Qh <= 0
        bTa_is_safe = bTah_is_safe.min(axis=-1)
        # bTah_A = jnp.where(bTa_is_safe[:, :, :, None], bTah_Al, jnp.zeros_like(bTah_Al))
        bTa_A = jnp.where(bTa_is_safe, bTa_Al, bTah_Ah.max(axis=-1))
        # bTa_A = bTah_A.sum(axis=-1) / (jnp.count_nonzero(bTah_A, axis=-1) + 1e-8)


        # update ppo
        def update_fn(carry, idx):
            critic, Vh, policy = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            critic, Vh, value_info = self.update_value(
                critic, Vh, rollout_batch, bT_Ql[idx], bTah_Qh[idx], rnn_states_V[idx], rnn_chunk_ids
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

    def update_policy(self, policy_train_state: TrainState, rollout: Rollout, bTa_A: Array, rnn_chunk_ids: Array):
        # all the agents share the same advantages
        # bTa_A = jnp.repeat(bT_A[:, :, None], self.n_agents, axis=-1)

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
            # bcTah_ratio = bcTa_ratio[:, :, :, :, None]
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
            rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        bcT_rollout = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout)  # (n_env, n_chunk, T, ...)
        rnn_state_inits = jnp.zeros_like(rnn_states[:, rnn_chunk_ids[:, 0]])  # (n_env, n_chunk, ...)
        bcT_Ql = bT_Ql[:, rnn_chunk_ids]
        bcTah_Qh = bTah_Qh[:, rnn_chunk_ids]

        def get_loss(critic_params, Vh_params):
            (bcTa_Vl, bcTah_Vh), _, _ = jax_vmap(jax_vmap(
                ft.partial(self.scan_value, critic_params=critic_params, Vh_params=Vh_params)))(
                bcT_rollout, rnn_state_inits, rnn_state_inits
            )
            loss_Vl = optax.l2_loss(bcTa_Vl.sum(-1), bcT_Ql).mean()
            loss_Vh = optax.l2_loss(bcTah_Vh, bcTah_Qh).mean()
            info = {
                'critic/loss': loss_Vl,
                'critic/loss_Vh': loss_Vh,
                # 'critic/unsafe': jnp.mean((bcTah_Vh.max(-1) > bcTa_Vl - bcT_rollout.zs.squeeze(-1))),
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