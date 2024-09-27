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

from .utils import compute_gae, HLGaussTransform, compute_dec_efocp_gae
from .base import Algorithm
from ..utils.typing import Action, Params, PRNGKey, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import tree_index, jax_vmap
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..trainer.utils import rollout as rollout_fn
from ..env.base import MultiAgentEnv
from ..algo.module.value import CostValueNet
from ..algo.module.policy import PPOPolicy


class InforMARL(Algorithm):

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
            Vl_gnn_layers: int = 2,
            gamma: float = 0.99,
            lr_actor: float = 3e-4,
            lr_Vl: float = 1e-3,
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
            hl_gauss_bins: int = 64,
            hl_gauss_smooth_ratio: float = 0.75,  # smooth_ratio = sigma / bin_width
            **kwargs
    ):
        super(InforMARL, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.cost_weight = cost_weight
        self.actor_gnn_layers = actor_gnn_layers
        self.Vl_gnn_layers = Vl_gnn_layers
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_Vl = lr_Vl
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.rnn_step = rnn_step
        self.use_lstm = use_lstm

        # HL Gauss config
        self.use_hl_gauss = use_hl_gauss
        self.hl_gauss_bins = hl_gauss_bins
        self.reward_min = env.reward_min
        self.reward_max = env.reward_max
        self.hl_gauss_sigma = hl_gauss_smooth_ratio * (self.reward_max - self.reward_min) / hl_gauss_bins
        self.Vl_hl_gauss_trans = HLGaussTransform(self.reward_min, self.reward_max, hl_gauss_bins, self.hl_gauss_sigma)

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
        self.nominal_z = jnp.zeros((1,))

        # set up PPO policy
        self.policy = PPOPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.actor_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm
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
        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_rnn_state = init_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        # initialize the policy
        policy_key, key = jr.split(key)
        self.policy_params = self.policy.dist.init(
            policy_key, nominal_graph, self.init_rnn_state, self.n_agents, self.nominal_z.repeat(self.n_agents, axis=0)
        )
        policy_optim = optax.adam(learning_rate=lr_actor)
        self.policy_optim = optax.apply_if_finite(policy_optim, 1_000_000)
        self.policy_train_state = TrainState.create(
            apply_fn=self.policy.sample_action,
            params=self.policy_params,
            tx=self.policy_optim
        )

        # set up PPO critic
        self.Vl = CostValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.Vl_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            use_ef=False,
            decompose=False,
            hl_gauss_bins=self.hl_gauss_bins if self.use_hl_gauss else None,
            relu_out=False
        )

        # initialize the rnn state
        rnn_state_key, key = jr.split(key)
        init_Vl_rnn_state = self.Vl.initialize_carry(rnn_state_key)  # (rnn_state_dim,)
        if type(init_Vl_rnn_state) is tuple:
            init_Vl_rnn_state = jnp.stack(init_Vl_rnn_state, axis=0)  # (n_carries, rnn_state_dim)
        else:
            init_Vl_rnn_state = init_Vl_rnn_state[None, :]
        # (n_rnn_layers, 1, n_carries, rnn_state_dim)
        self.init_Vl_rnn_state = init_Vl_rnn_state[None, :, :].repeat(self.rnn_layers, axis=0)[:, None, :, :]

        # initialize the critic
        Vl_key, key = jr.split(key)
        Vl_params = self.Vl.net.init(Vl_key, nominal_graph, self.init_Vl_rnn_state, self.n_agents, self.nominal_z)
        Vl_optim = optax.adam(learning_rate=lr_Vl)
        self.Vl_optim = optax.apply_if_finite(Vl_optim, 1_000_000)
        self.Vl_train_state = TrainState.create(
            apply_fn=self.Vl.get_value,
            params=Vl_params,
            tx=self.Vl_optim
        )

        # set up key
        self.key = key

        def rollout_fn_single_(cur_params, cur_key):
            return rollout_fn(self._env,
                              ft.partial(self.step, params=cur_params),
                              self.init_rnn_state,
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
            'Vl_gnn_layers': self.Vl_gnn_layers,
            'gamma': self.gamma,
            'lr_actor': self.lr_actor,
            'lr_Vl': self.lr_Vl,
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
            'use_hl_gauss': self.use_hl_gauss,
            'hl_gauss_bins': self.hl_gauss_bins,
            'hl_gauss_smooth_ratio': self.hl_gauss_sigma * self.hl_gauss_bins
        }

    @property
    def params(self) -> Params:
        return {
            "policy": self.policy_train_state.params,
            "Vl": self.Vl_train_state.params,
        }

    def act(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            params: Optional[Params] = None,
            z: Optional[Array] = None
    ) -> [Action, Array]:
        if params is None:
            params = self.params
        action, rnn_state = self.policy.get_action(params["policy"], graph, rnn_state, z)
        return action, rnn_state

    def step(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            key: PRNGKey,
            params: Optional[Params] = None,
            z: Optional[Array] = None
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
            Vl_train_state, policy_train_state, update_info = self.update_inner(
                self.Vl_train_state, self.policy_train_state, rollout, batch_idx, rnn_chunk_ids
            )
            self.Vl_train_state = Vl_train_state
            self.policy_train_state = policy_train_state
        return update_info

    def scan_Vl(
            self, rollout: Rollout, init_Vl_rnn_state: Array, Vl_params: Params
    ) -> Tuple[Array, Array, Array]:
        T_graphs = rollout.graph  # (T, ...)

        def body_(rnn_state, graph):
            value, new_rnn_state = self.Vl.get_value(Vl_params, graph, rnn_state)
            return new_rnn_state, (value, rnn_state)

        final_rnn_state, (T11_Vl, T_rnn_states) = jax.lax.scan(body_, init_Vl_rnn_state, T_graphs)
        T_Vl = T11_Vl.squeeze(1).squeeze(1)

        return T_Vl, T_rnn_states, final_rnn_state

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            Vl_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, TrainState, dict]:
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

        b_final_Vl = jax.vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape[:2] == (b, T + 1)

        if self.use_hl_gauss:
            bTp1_Vl = jax_vmap(jax_vmap(self.Vl_hl_gauss_trans.logits2value))(bTp1_Vl)

        # calculate GAE
        bT_targets, bT_A = compute_gae(
            values=bTp1_Vl[:, :-1],
            rewards=rollout.rewards - self.cost_weight * jnp.maximum(rollout.costs, 0.0).sum(axis=-1).sum(axis=-1),
            dones=rollout.dones,
            next_values=bTp1_Vl[:, 1:],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        # bTp1ah_Vh = bTp1_Vl[:, :, None, None].repeat(self.n_agents, axis=-2).repeat(rollout.costs.shape[-1], axis=-1)
        # bTah_Qh, bT_Ql, bTa_Q = jax.vmap(
        #     ft.partial(compute_dec_efocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        # )(Tah_hs=rollout.costs,
        #   T_l=-rollout.rewards,
        #   T_z=rollout.zs.squeeze(-1)[:, :, 0],
        #   Tp1ah_Vh=bTp1ah_Vh,
        #   Tp1_Vl=bTp1_Vl)
        # bT_Al = bT_Ql - bT_Vl
        # bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        # bTa_A = -bT_Al[:, :, None].repeat(self.n_agents, axis=-1)

        # all the agents share the same GAEs
        bTa_A = jnp.repeat(bT_A[:, :, None], self.n_agents, axis=-1)

        # ppo update
        def update_fn(carry, idx):
            Vl_model, policy_model = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model, rollout_batch, bT_targets[idx], bT_Vl_rnn_states[idx], rnn_chunk_ids)
            policy_model, policy_info = self.update_policy(policy_model, rollout_batch, bTa_A[idx], rnn_chunk_ids)
            return (Vl_model, policy_model), (Vl_info | policy_info)

        (Vl_train_state, policy_train_state), info = lax.scan(
            update_fn, (Vl_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)

        return Vl_train_state, policy_train_state, info

    def update_Vl(
            self,
            Vl_train_state: TrainState,
            rollout: Rollout,
            bT_targets: Array,
            bT_rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        bcT_rollout = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout)
        bcT_targets = bT_targets[:, rnn_chunk_ids]
        bc_rnn_state_inits = jnp.zeros_like(bT_rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        def get_loss_(params):
            bcT_Vl, bcT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_Vl,
                           Vl_params=params)
            ))(bcT_rollout, bc_rnn_state_inits)
            loss_Vl = optax.l2_loss(bcT_Vl, bcT_targets).mean()
            return loss_Vl

        def get_loss_hl_gauss_(params):
            bcT_Vl, bcT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_Vl,
                           Vl_params=params)
            ))(bcT_rollout, bc_rnn_state_inits)
            bcT_targets_dist = jax_vmap(jax_vmap(jax_vmap(self.Vl_hl_gauss_trans.value2dist)))(bcT_targets)
            loss_Vl = optax.softmax_cross_entropy(bcT_Vl, bcT_targets_dist).mean()
            return loss_Vl

        if self.use_hl_gauss:
            loss, grad = jax.value_and_grad(get_loss_hl_gauss_)(Vl_train_state.params)
        else:
            loss, grad = jax.value_and_grad(get_loss_)(Vl_train_state.params)
        critic_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        critic_train_state = Vl_train_state.apply_gradients(grads=grad)
        return critic_train_state, {'Vl/loss': loss,
                                    'Vl/grad_norm': grad_norm,
                                    'Vl/has_nan': critic_has_nan,
                                    'Vl/max_target': jnp.max(bT_targets),
                                    'Vl/min_target': jnp.min(bT_targets)}

    def scan_eval_action(
            self,
            T_graphs: GraphsTuple,
            Ta_actions: Action,
            init_rnn_state: Array,
            T_action_keys: PRNGKey,
            actor_params: Params
    ) -> Tuple[Array, Array, Array, Array]:
        def body_(rnn_state, inp):
            graph, action, key = inp
            log_pi, entropy, new_rnn_state = self.policy.eval_action(actor_params, graph, action, rnn_state, key)
            return new_rnn_state, (log_pi, entropy, rnn_state)

        final_rnn_state, outputs = jax.lax.scan(body_, init_rnn_state, (T_graphs, Ta_actions, T_action_keys))
        Ta_log_pis, Ta_entropies, T_rnn_states = outputs

        return Ta_log_pis, Ta_entropies, T_rnn_states, final_rnn_state

    def update_policy(
            self, policy_train_state: TrainState, rollout: Rollout, bTa_A: Array, rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        # divide the rollout into chunks (n_env, n_chunks, T, ...)
        bcT_graph = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.graph)
        bcTa_action = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.actions)
        bcTa_log_pis_old = jax.tree_map(lambda x: x[:, rnn_chunk_ids], rollout.log_pis)
        bcTa_A = jax.tree_map(lambda x: x[:, rnn_chunk_ids], bTa_A)
        bc_rnn_state_inits = jnp.zeros_like(rollout.rnn_states[:, rnn_chunk_ids[:, 0]])  # use zeros rnn_state as init

        action_key = jr.fold_in(self.key, policy_train_state.step)
        action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))
        bcT_action_keys = jax.tree_map(lambda x: x[:, rnn_chunk_ids], action_keys)

        def get_loss_(params):
            bcTa_log_pis, bcTa_policy_entropy, bcT_rnn_states, final_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_eval_action,
                           actor_params=params)
            ))(bcT_graph, bcTa_action, bc_rnn_state_inits, bcT_action_keys)

            bcTa_ratio = jnp.exp(bcTa_log_pis - bcTa_log_pis_old)
            loss_policy1 = -bcTa_ratio * bcTa_A
            loss_policy2 = -jnp.clip(bcTa_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * bcTa_A
            clip_frac = jnp.mean(loss_policy2 > loss_policy1)
            loss_policy = jnp.maximum(loss_policy1, loss_policy2).mean()
            total_entropy = bcTa_policy_entropy.mean()
            policy_loss = loss_policy - self.coef_ent * total_entropy
            total_variation_dist = 0.5 * jnp.mean(jnp.abs(bcTa_ratio - 1.0))
            return policy_loss, {'policy/clip_frac': clip_frac,
                                 'policy/entropy': bcTa_policy_entropy.mean(),
                                 'policy/total_variation_dist': total_variation_dist}

        (loss, info), grad = jax.value_and_grad(get_loss_, has_aux=True)(policy_train_state.params)
        policy_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)

        # clip grad
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)

        # update policy
        policy_train_state = policy_train_state.apply_gradients(grads=grad)

        # get info
        info = {
                   'policy/loss': loss,
                   'policy/grad_norm': grad_norm,
                   'policy/has_nan': policy_has_nan,
                   'policy/log_pi_min': rollout.log_pis.min()
               } | info

        return policy_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.Vl_train_state.params, open(os.path.join(model_dir, 'Vl.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.Vl_train_state = \
            self.Vl_train_state.replace(params=pickle.load(open(os.path.join(path, 'Vl.pkl'), 'rb')))
