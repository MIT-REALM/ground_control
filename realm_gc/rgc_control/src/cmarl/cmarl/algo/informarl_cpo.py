import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle
import flax.linen as nn

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax
from equinox.debug import breakpoint_if

from ..utils.typing import Action, Params, PRNGKey, Array, List
from ..utils.graph import GraphsTuple
from ..utils.utils import merge01, jax_vmap, tree_merge, tree_index
from ..trainer.data import Rollout
from ..trainer.buffer import ReplayBuffer
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..trainer.utils import rollout as rollout_fn
from ..env.base import MultiAgentEnv
from ..algo.module.value import CostValueNet
from ..algo.module.policy import PPOPolicy
from .utils import compute_gae, compute_dec_efocp_gae
from .base import Algorithm
from .informarl import InforMARL
from .informarl_lagr import InforMARLLagr


class InforMARLCPO(InforMARLLagr):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            actor_gnn_layers: int = 2,
            Vl_gnn_layers: int = 2,
            Vh_gnn_layers: int = 1,
            gamma: float = 0.99,
            lr_actor: float = 3e-4,
            lr_Vl: float = 1e-3,
            lr_Vh: float = 1e-3,
            batch_size: int = 8192,
            epoch_ppo: int = 1,
            clip_eps: float = 0.25,
            gae_lambda: float = 0.95,
            coef_ent: float = 1e-2,
            max_grad_norm: float = 2.0,
            seed: int = 0,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            rnn_step: int = 16,
            use_lstm: bool = False,
            fraction_coef: float = 0.27,
            kl_threshold: float = 0.0065,
            **kwargs
    ):
        super(InforMARLCPO, self).__init__(
            env, node_dim, edge_dim, state_dim, action_dim, n_agents, actor_gnn_layers, Vl_gnn_layers, Vh_gnn_layers,
            gamma, lr_actor, lr_Vl, lr_Vh, batch_size, epoch_ppo, clip_eps, gae_lambda, coef_ent, max_grad_norm, seed,
            use_rnn, rnn_layers, rnn_step, use_lstm
        )

        # set hyperparameters
        self.fraction_coef = fraction_coef
        self.kl_threshold = kl_threshold

    @property
    def config(self) -> dict:
        return super().config | {
            "fraction_coef": self.fraction_coef,
            "kl_threshold": self.kl_threshold
        }

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
            Vl_train_state, Vh_train_state, policy_train_state, update_info = self.update_inner(
                self.Vl_train_state,
                self.Vh_train_state,
                self.policy_train_state,
                rollout,
                batch_idx,
                rnn_chunk_ids
            )
            self.Vl_train_state = Vl_train_state
            self.Vh_train_state = Vh_train_state
            self.policy_train_state = policy_train_state
        return update_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            Vl_train_state: TrainState,
            Vh_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
    ) -> Tuple[TrainState, TrainState, TrainState, dict]:
        # rollout: (n_env, T, n_agent, ...)
        b, T, a, _ = rollout.zs.shape

        # calculate Vl
        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(self.scan_Vl,
                       init_Vl_rnn_state=self.init_Vl_rnn_state,
                       Vl_params=Vl_train_state.params)
        )(rollout)

        def final_Vl_fn_(graph, rnn_state_Vl):
            Vl, _ = self.Vl.get_value(Vl_train_state.params, tree_index(graph, -1), rnn_state_Vl)
            return Vl.squeeze(-1).squeeze(-1)

        b_final_Vl = jax.vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape == (b, T + 1)

        # calculate Vh
        bTah_Vh, bT_Vh_rnn_states, final_Vh_rnn_states = jax.vmap(
            ft.partial(self.scan_Vh,
                       init_rnn_state=self.init_Vh_rnn_state,
                       Vh_params=Vh_train_state.params)
        )(rollout)

        def final_Vh_fn_(graph, rnn_state_Vh):
            Vh, _ = self.Vh.get_value(Vh_train_state.params, tree_index(graph, -1), rnn_state_Vh)
            return Vh

        bah_final_Vh = jax.vmap(final_Vh_fn_)(rollout.next_graph, final_Vh_rnn_states)
        bTp1ah_Vh = jnp.concatenate([bTah_Vh, bah_final_Vh[:, None]], axis=1)
        assert bTp1ah_Vh.shape == (b, T + 1, self.n_agents, self._env.n_cost)

        # calculate Dec-EFOCP GAE
        bTah_Qh, bT_Ql, bTa_Q = jax.vmap(
            ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=rollout.costs,
          T_l=-rollout.rewards,
          T_z=rollout.zs.squeeze(-1)[:, :, 0],
          Tp1ah_Vh=bTp1ah_Vh,
          Tp1_Vl=bTp1_Vl)

        # calculate advantages and normalize
        # cost advantage
        bT_Al: Array = bT_Ql - bT_Vl
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        bTa_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)
        bTa_Al = -bTa_Al
        assert bTa_Al.shape == (b, T, self.n_agents)

        # constraint advantage
        bTah_Ah: Array = bTah_Qh - bTah_Vh
        bTah_Ah = (bTah_Ah - bTah_Ah.mean(axis=1, keepdims=True)) / (bTah_Ah.std(axis=1, keepdims=True) + 1e-8)
        bTah_Ah = -bTah_Ah
        assert bTah_Ah.shape == (b, T, self.n_agents, self._env.n_cost)

        # trpo update
        def update_fn(carry, idx):
            Vl_model, Vh_model, policy_model = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model, rollout_batch, bT_Ql[idx], bT_Vl_rnn_states[idx], rnn_chunk_ids)
            Vh_model, Vh_info = self.update_Vh(
                Vh_model, rollout_batch, bTah_Qh[idx], bT_Vh_rnn_states[idx], rnn_chunk_ids)
            policy_model, policy_info = self.update_policy_trpo(
                policy_model, rollout_batch, bTa_Al[idx], bTah_Ah[idx], rnn_chunk_ids)

            return (Vl_model, Vh_model, policy_model), (Vl_info | Vh_info | policy_info)

        (Vl_train_state, Vh_train_state, policy_train_state, ah_lagr), update_info = jax.lax.scan(
            update_fn, (Vl_train_state, Vh_train_state, policy_train_state), batch_idx)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], update_info)

        return Vl_train_state, Vh_train_state, policy_train_state, info

    def kl_divergence(
            self,
            T_graphs: GraphsTuple,
            Ta_actions: Action,
            init_rnn_state: Array,
            T_action_keys: PRNGKey,
            actor_params_new: Params,
            actor_params_old: Params
    ):
        _, _, _, mean_new, std_new, _ = self.scan_eval_action_trpo(
            T_graphs, Ta_actions, init_rnn_state, T_action_keys, actor_params_new)
        _, _, _, mean_old, std_old, _ = self.scan_eval_action_trpo(
            T_graphs, Ta_actions, init_rnn_state, T_action_keys, actor_params_old)

        kl = jnp.sum(
            jnp.log(std_old) - jnp.log(std_new) +
            (std_old ** 2 + (mean_old - mean_new) ** 2) / (2 * std_new ** 2 + 1e-8) - 0.5
        )

        return kl

    def scan_eval_action_trpo(
            self,
            T_graphs: GraphsTuple,
            Ta_actions: Action,
            init_rnn_state: Array,
            T_action_keys: PRNGKey,
            actor_params: Params
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:
        def body_(rnn_state, inp):
            graph, action, key = inp
            log_pi, entropy, new_rnn_state, mean, std = self.policy.eval_action_trpo(
                actor_params, graph, action, rnn_state, key)
            return new_rnn_state, (log_pi, entropy, rnn_state, mean, std)

        final_rnn_state, outputs = jax.lax.scan(body_, init_rnn_state, (T_graphs, Ta_actions, T_action_keys))
        Ta_log_pis, Ta_entropies, T_rnn_states, T_mean, T_std = outputs

        return Ta_log_pis, Ta_entropies, T_rnn_states, T_mean, T_std, final_rnn_state

    def update_policy_trpo(
            self, policy_train_state: TrainState, rollout: Rollout, bTa_Al: Array, bTah_Ah: Array, rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        # divide the rollout into chunks (n_env, n_chunks, T, ...)
        bcT_graph = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
        bcTa_action = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.actions)
        bcTa_log_pis_old = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.log_pis)
        bcTa_Al = jax.tree_map(lambda x: x[:, rnn_chunk_ids], bTa_Al)
        bcTah_Ah = jax.tree_map(lambda x: x[:, rnn_chunk_ids], bTah_Ah)
        bc_rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        bcT_action_keys = jax.tree_map(lambda x: x[:, rnn_chunk_ids], action_keys)

        def get_cost_loss_(params):
            bcTa_log_pis, bcTa_policy_entropy, bcT_rnn_states, final_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action,
                           actor_params=params)
            ))(bcT_graph, bcTa_action, bc_rnn_state_inits, bcT_action_keys)

            bcTa_ratio = jnp.exp(bcTa_log_pis - bcTa_log_pis_old)
            loss = jnp.mean(-bcTa_ratio * bcTa_Al)

            return loss

        def get_constraint_loss_(params):
            bcTa_log_pis, bcTa_policy_entropy, bcT_rnn_states, final_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action,
                           actor_params=params)
            ))(bcT_graph, bcTa_action, bc_rnn_state_inits, bcT_action_keys)

            bcTa_ratio = jnp.exp(bcTa_log_pis - bcTa_log_pis_old)
            loss = jnp.mean(-bcTa_ratio[:, :, :, :, None] * bcTah_Ah)  # todo: can we combine the two functions?

            return loss

        cost_loss, cost_loss_grad = jax.value_and_grad(get_cost_loss_)(policy_train_state.params)
        constraint_loss, constraint_loss_grad = jax.value_and_grad(get_constraint_loss_)(policy_train_state.params)

        aaa = 0
