import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax
from tqdm import trange, tqdm

from .utils import compute_dec_efocp_gae
from .informarl_lagr import InforMARLLagr
from ..utils.typing import Action, Params, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import jax_vmap, tree_index, rep_vmap
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..trainer.utils import test_rollout as det_rollout_fn
from ..env.base import MultiAgentEnv
from ..algo.module.value import CostValueNet


class GCBFCRPO(InforMARLLagr):

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
            use_hl_gauss: bool = False,
            Vh_use_hl_gauss: bool = False,
            hl_gauss_bins: int = 64,
            hl_gauss_smooth_ratio: float = 0.75,  # smooth_ratio = sigma / bin_width
            alpha: float = 10.0,
            cbf_eps: float = 1e-2,
            cbf_weight: float = 1.0,
            lr_schedule: bool = False,
            train_steps: int = 1e5,
            gamma_schedule: bool = False,
            cbf_schedule: bool = False,
            alpha_schedule: bool = False,
            **kwargs
    ):
        super(GCBFCRPO, self).__init__(
            env, node_dim, edge_dim, state_dim, action_dim, n_agents, actor_gnn_layers, Vl_gnn_layers, Vh_gnn_layers,
            gamma, lr_actor, lr_Vl, lr_Vh, batch_size, epoch_ppo, clip_eps, gae_lambda, coef_ent, max_grad_norm, seed,
            use_rnn, rnn_layers, rnn_step, use_lstm, use_hl_gauss, hl_gauss_bins, hl_gauss_smooth_ratio
        )

        # set hyperparameters
        self.alpha = alpha
        self.cbf_eps = cbf_eps
        self.cbf_weight = cbf_weight
        self.Vh_use_hl_gauss = Vh_use_hl_gauss
        self.Vh_init_step = 1000
        self.gamma_schedule = gamma_schedule
        self.cbf_schedule = cbf_schedule
        self.alpha_shchedule = alpha_schedule
        # self.gamma_Vh = gamma

        if self.cbf_schedule:
            if cbf_weight == 1.0:
                boundaries_and_scales={
                    int(train_steps * 0.25): 2,
                    int(train_steps * 0.5): 2,
                    int(train_steps * 0.75): 2,
                }
            else:
                boundaries_and_scales={
                    int(train_steps * 0.25): 1.5,
                    int(train_steps * 0.5): 1.5,
                    int(train_steps * 0.75): 2,
                }

            self.cbf_schedule_fn = optax.piecewise_constant_schedule(
                init_value=cbf_weight,
                boundaries_and_scales=boundaries_and_scales
            )
            # self.cbf_schedule_1 = optax.constant_schedule(cbf_weight)
            # self.cbf_schedule_2 = optax.linear_schedule(
            #     init_value=cbf_weight,
            #     end_value=cbf_weight * 4,
            #     transition_steps=train_steps // 2
            # )
            # self.cbf_schedule_fn = optax.join_schedules(
            #    [self.cbf_schedule_1, self.cbf_schedule_2], [0, train_steps // 2]
            # )
            # self.cbf_schedule_fn = optax.piecewise_constant_schedule(
            #     init_value=cbf_weight,
            #     boundaries_and_scales={
            #         int(train_steps * 0.5): 2,
            #         int(train_steps * 0.8): 2,
            #     }
            # )

        if self.alpha_shchedule:
            self.alpha_schedule_fn = optax.piecewise_constant_schedule(
                init_value=alpha,
                boundaries_and_scales={
                    int(train_steps * 0.7): 2,
                    int(train_steps * 0.9): 2,
                }
            )
            # self.alpha_schedule_1 = optax.constant_schedule(alpha)
            # self.alpha_schedule_2 = optax.linear_schedule(
            #     init_value=alpha,
            #     end_value=alpha * 4,
            #     transition_steps=train_steps // 2
            # )
            # self.alpha_schedule_fn = optax.join_schedules(
            #     [self.alpha_schedule_1, self.alpha_schedule_2], [0, train_steps // 2]
            # )

        if self.gamma_schedule:
            self.gamma_schedule_fn = optax.linear_schedule(
                init_value=gamma * 0.01,
                end_value=gamma,
                transition_steps=train_steps // 2
            )
        else:
            self.gamma_schedule_fn = optax.constant_schedule(gamma)

        # set up constraint value net
        self.Vh = CostValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            n_out=env.n_cost,
            use_rnn=self.use_rnn,
            gnn_layers=Vh_gnn_layers,
            gnn_out_dim=64,
            use_ef=False,
            use_lstm=False,
            decompose=True,
            use_global_info=False,  # use only local information
            hl_gauss_bins=hl_gauss_bins if self.Vh_use_hl_gauss else None,
        )

        Vh_key, self.key = jr.split(self.key)
        Vh_params = self.Vh.net.init(Vh_key, self.nominal_graph, self.init_rnn_state, self.n_agents, self.nominal_z)
        Vh_optim = optax.adam(learning_rate=lr_Vh)
        self.Vh_optim = optax.apply_if_finite(Vh_optim, 1_000_000)
        self.Vh_train_state = TrainState.create(
            apply_fn=self.Vh.get_value,
            params=Vh_params,
            tx=self.Vh_optim
        )

        # rollout function with deterministic policy
        def det_rollout_fn_single_(cur_params, cur_key):
            return det_rollout_fn(self._env,
                                  ft.partial(self.act, params=cur_params),
                                  self.init_rnn_state,
                                  cur_key)

        def det_rollout_fn_(cur_params, cur_keys):
            return jax.vmap(ft.partial(det_rollout_fn_single_, cur_params))(cur_keys)

        self.det_rollout_fn = jax.jit(det_rollout_fn_)

        if lr_schedule:
            # lr_actor_schedule = optax.linear_schedule(lr_actor, lr_actor * 1e-2, train_steps)
            lr_actor_schedule = optax.piecewise_constant_schedule(
                init_value=lr_actor,
                boundaries_and_scales={
                    int(train_steps * 0.1): 1 / 3,
                    int(train_steps * 0.2): 1 / 3,
                    int(train_steps * 0.3): 1 / 3
                }
            )
            policy_optim = optax.adam(learning_rate=lr_actor_schedule)
            self.policy_optim = optax.apply_if_finite(policy_optim, 1_000_000)
            self.policy_train_state = TrainState.create(
                apply_fn=self.policy.sample_action,
                params=self.policy_params,
                tx=self.policy_optim
            )

    @property
    def config(self) -> dict:
        return super().config | {
            'alpha': self.alpha,
            'cbf_eps': self.cbf_eps,
            'cbf_weight': self.cbf_weight,
            'Vh_use_hl_gauss': self.Vh_use_hl_gauss
        }

    def get_Vh(
            self, graph: GraphsTuple, rnn_state: Array, params: Optional[Params] = None
    ) -> Array:
        if params is None:
            params = self.params
        Vh, _ = self.Vh.get_value(params["Vh"], graph, rnn_state)
        return Vh

    @ft.partial(jax.jit, static_argnums=(0,))
    def initialize_Vh(
            self, policy_train_state: TrainState, Vh_train_state: TrainState, det_rollout: Rollout
    ) -> Tuple[TrainState, dict]:
        # calculate Vh for deterministic policy
        bTah_Vh_det = jax.vmap(jax.vmap(ft.partial(
            self.get_Vh, params={'Vh': Vh_train_state.params})))(det_rollout.graph, det_rollout.rnn_states)

        def final_Vh_fn_(graph, rnn_state):
            _, final_rnn_state = self.act(tree_index(graph, -1), rnn_state[-1], {'policy': policy_train_state.params})
            return self.get_Vh(tree_index(graph, -1), final_rnn_state, {'Vh': Vh_train_state.params})

        final_Vh_det = jax.vmap(final_Vh_fn_)(det_rollout.next_graph, det_rollout.rnn_states)
        bTp1ah_Vh_det = jnp.concatenate([bTah_Vh_det, final_Vh_det[:, None]], axis=1)

        if self.Vh_use_hl_gauss:
            bTp1ah_Vh_det = rep_vmap(self.Vh_hl_gauss_trans.logits2value, rep=4)(bTp1ah_Vh_det)

        # calculate Qh for deterministic policy
        bTah_Qh_det, _, _ = jax.vmap(
            ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=det_rollout.costs,
          T_l=-det_rollout.rewards,
          T_z=det_rollout.zs.squeeze(-1)[:, :, 0],
          Tp1ah_Vh=bTp1ah_Vh_det,
          Tp1_Vl=bTp1ah_Vh_det.max(axis=-1).max(axis=-1))

        rnn_chunk_ids = jnp.arange(det_rollout.dones.shape[1])
        rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, det_rollout.dones.shape[1] // self.rnn_step))
        Vh_train_state, Vh_info = self.update_Vh(
            Vh_train_state, det_rollout, bTah_Qh_det, det_rollout.rnn_states, rnn_chunk_ids)

        return Vh_train_state, Vh_info

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        # learn initial Vh
        # if step == 0:
        #     tqdm.write("Initializing Vh")
        #     for _ in trange(0, self.Vh_init_step, ncols=80):
        #         b_key = jr.split(key, rollout.dones.shape[0])
        #         det_rollout = self.det_rollout_fn(self.params, b_key)
        #         self.Vh_train_state, Vh_info = self.initialize_Vh(
        #             self.policy_train_state, self.Vh_train_state, det_rollout)

        use_safety = jnp.array([step > 0])

        # get rollout with deterministic policy
        b_key = jr.split(key, rollout.dones.shape[0])
        det_rollout = self.det_rollout_fn(self.params, b_key)

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
                det_rollout,
                batch_idx,
                rnn_chunk_ids,
                use_safety,
                jnp.array(step)
            )
            self.Vl_train_state = Vl_train_state
            self.Vh_train_state = Vh_train_state
            self.policy_train_state = policy_train_state
        return update_info

    def scan_act(self, rollout: Rollout, init_rnn_state: Array, actor_params: Params) -> Tuple[Action, Array, Array]:
        T_graph = rollout.graph
        Ta_z = rollout.zs

        def body_(rnn_state, inp):
            graph, z = inp
            action, new_rnn_state = self.act(graph, z, rnn_state, actor_params)
            return new_rnn_state, (action, rnn_state)

        final_rnn_state, outputs = jax.lax.scan(body_, init_rnn_state, (T_graph, Ta_z))
        Ta_actions, T_rnn_states = outputs

        return Ta_actions, T_rnn_states, final_rnn_state

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            Vl_train_state: TrainState,
            Vh_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            det_rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
            use_safety: Array,
            step: Array
    ) -> Tuple[TrainState, TrainState, TrainState, dict]:
        # rollout: (b, T, a, ...)
        b, T, a, _ = rollout.zs.shape

        # calculate Vl
        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(self.scan_Vl,
                       init_Vl_rnn_state=self.init_Vl_rnn_state,
                       Vl_params=Vl_train_state.params)
        )(rollout)

        def final_Vl_fn_(graph, rnn_state):
            Vl, _ = self.Vl.get_value(Vl_train_state.params, tree_index(graph, -1), rnn_state)
            return Vl.squeeze(0).squeeze(0)

        b_final_Vl = jax_vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape[:2] == (b, T + 1)

        if self.use_hl_gauss:
            bTp1_Vl = rep_vmap(self.Vl_hl_gauss_trans.logits2value, rep=2)(bTp1_Vl)
            bT_Vl = rep_vmap(self.Vl_hl_gauss_trans.logits2value, rep=2)(bT_Vl)

        # calculate Vh
        bTah_Vh = jax.vmap(jax.vmap(ft.partial(
            self.get_Vh, params={'Vh': Vh_train_state.params})))(rollout.graph, rollout.rnn_states)

        def final_Vh_fn_(graph, rnn_state):
            _, final_rnn_state = self.act(tree_index(graph, -1), rnn_state[-1], {'policy': policy_train_state.params})
            return self.get_Vh(tree_index(graph, -1), final_rnn_state, {'Vh': Vh_train_state.params})

        final_Vh = jax.vmap(final_Vh_fn_)(rollout.next_graph, rollout.rnn_states)

        bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)
        assert bTp1ah_Vh.shape[:4] == (b, T + 1, a, self._env.n_cost)

        if self.Vh_use_hl_gauss:
            bTp1ah_Vh = rep_vmap(self.Vh_hl_gauss_trans.logits2value, rep=4)(bTp1ah_Vh)
            bTah_Vh = rep_vmap(self.Vh_hl_gauss_trans.logits2value, rep=4)(bTah_Vh)

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
        bT_Al = bT_Ql - bT_Vl
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        bTa_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)

        # CBF advantage
        if self.alpha_shchedule:
            alpha = self.alpha_schedule_fn(step)
        else:
            alpha = self.alpha
        bTah_cbf_deriv = (bTp1ah_Vh[:, 1:] - bTah_Vh) / self._env.dt + alpha * bTah_Vh
        bTah_Acbf = jnp.maximum(bTah_cbf_deriv + self.cbf_eps, 0)
        # bTah_Acbf = (bTah_Acbf - bTah_Acbf.mean(axis=1, keepdims=True)) / (bTah_Acbf.std(axis=1, keepdims=True) + 1e-8)
        # bTah_Acbf = bTah_Acbf / (bTah_Acbf.std(axis=1, keepdims=True) + 1e-8)

        # merge advantage
        bTa_is_safe = (bTah_cbf_deriv <= 0).min(axis=-1)
        safe_data = bTa_is_safe.mean()
        bTa_A = jnp.where(bTa_is_safe, bTa_Al, jnp.zeros_like(bTa_Al))
        if self.cbf_schedule:
            bTa_A += bTah_Acbf.max(axis=-1) * self.cbf_schedule_fn(step)
        else:
            bTa_A += bTah_Acbf.max(axis=-1) * self.cbf_weight

        # reverse advantage
        bTa_A = jnp.where(use_safety, -bTa_A, -bTa_Al)
        # bTa_A = -bTa_A

        # calculate Vh for deterministic policy
        bTah_Vh_det = jax.vmap(jax.vmap(ft.partial(
            self.get_Vh, params={'Vh': Vh_train_state.params})))(det_rollout.graph, det_rollout.rnn_states)
        final_Vh_det = jax.vmap(final_Vh_fn_)(det_rollout.next_graph, det_rollout.rnn_states)
        bTp1ah_Vh_det = jnp.concatenate([bTah_Vh_det, final_Vh_det[:, None]], axis=1)

        if self.Vh_use_hl_gauss:
            bTp1ah_Vh_det = rep_vmap(self.Vh_hl_gauss_trans.logits2value, rep=4)(bTp1ah_Vh_det)

        # calculate Qh for deterministic policy
        bTah_Qh_det, _, _ = jax.vmap(
            ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma_schedule_fn(step), gae_lambda=self.gae_lambda)
        )(Tah_hs=det_rollout.costs,
          T_l=-det_rollout.rewards,
          T_z=det_rollout.zs.squeeze(-1)[:, :, 0],
          Tp1ah_Vh=bTp1ah_Vh_det,
          Tp1_Vl=bTp1_Vl)

        # ppo update
        def update_fn(carry, idx):
            Vl_model, Vh_model, policy_model = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            det_rollout_batch = jtu.tree_map(lambda x: x[idx], det_rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model, rollout_batch, bT_Ql[idx], bT_Vl_rnn_states[idx], rnn_chunk_ids)
            Vh_model, Vh_info = self.update_Vh(
                Vh_model, det_rollout_batch, bTah_Qh_det[idx], rollout.rnn_states[idx], rnn_chunk_ids)
            policy_model, policy_info = self.update_policy(policy_model, rollout_batch, bTa_A[idx], rnn_chunk_ids)
            return (Vl_model, Vh_model, policy_model), (Vl_info | Vh_info | policy_info)

        (Vl_train_state, Vh_train_state, policy_train_state), info = lax.scan(
            update_fn, (Vl_train_state, Vh_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info) | {'eval/safe_data': safe_data}

        return Vl_train_state, Vh_train_state, policy_train_state, info

    def update_Vh(
            self,
            Vh_train_state: TrainState,
            det_rollout: Rollout,
            bTah_Qh_det: Array,
            bT_rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        bcT_rollout = jax.tree_map(lambda x: x[:, rnn_chunk_ids], det_rollout)  # (n_env, n_chunk, T, ...)
        bcTah_Qh_det = bTah_Qh_det[:, rnn_chunk_ids]

        def get_loss(Vh_params):
            bcTah_Vh_det = jax.vmap(jax.vmap(jax.vmap(ft.partial(self.get_Vh, params={'Vh': Vh_params}))))(
                bcT_rollout.graph, bcT_rollout.rnn_states)
            loss_Vh = optax.l2_loss(bcTah_Vh_det, bcTah_Qh_det).mean()
            info = {
                'Vh/loss_Vh': loss_Vh
            }
            return loss_Vh, info

        def get_loss_hl_gauss_(Vh_params):
            bcTah_Vh_det = jax.vmap(jax.vmap(jax.vmap(ft.partial(self.get_Vh, params={'Vh': Vh_params}))))(
                bcT_rollout.graph, bcT_rollout.rnn_states)
            bcTah_Qh_det_dist = rep_vmap(self.Vh_hl_gauss_trans.value2dist, rep=5)(bcTah_Qh_det)
            loss_Vh = optax.softmax_cross_entropy(bcTah_Vh_det, bcTah_Qh_det_dist).mean()
            info = {
                'Vh/loss_Vh': loss_Vh
            }
            return loss_Vh, info

        if self.Vh_use_hl_gauss:
            grad_Vh, Vh_info = jax.grad(get_loss_hl_gauss_, has_aux=True)(Vh_train_state.params)
        else:
            grad_Vh, Vh_info = jax.grad(get_loss, has_aux=True)(Vh_train_state.params)
        grad_Vh_has_nan = has_any_nan_or_inf(grad_Vh).astype(jnp.float32)
        grad_Vh, grad_Vh_norm = compute_norm_and_clip(grad_Vh, self.max_grad_norm)
        Vh_train_state = Vh_train_state.apply_gradients(grads=grad_Vh)

        return Vh_train_state, Vh_info | {'Vh/grad_Vh_norm': grad_Vh_norm, 'Vh/grad_Vh_has_nan': grad_Vh_has_nan}
