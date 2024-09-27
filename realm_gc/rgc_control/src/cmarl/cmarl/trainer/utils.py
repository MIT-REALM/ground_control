import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import functools as ft
import seaborn as sns
import optax
import os

from typing import Callable, TYPE_CHECKING, Optional
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey, Array
from ..utils.graph import GraphsTuple
from .data import Rollout


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        gamma: float,
        init_graph: Optional[GraphsTuple] = None
) -> Rollout:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, Array, RNN_States, PRNGKey] -> [Action, LogPi, RNN_States]
    init_rnn_state: Array
    key: PRNGKey
    gamma: float, discount factor

    Returns
    -------
    data: Rollout
    """
    key_x0, key_z0, key = jax.random.split(key, 3)
    if init_graph is None:
        init_graph = env.reset(key_x0)
    # z0 = jax.random.uniform(key_z0, (env.num_agents, 1), minval=-env.reward_max, maxval=-env.reward_min)
    z0 = jax.random.uniform(key_z0, (1, 1), minval=-env.reward_max, maxval=-env.reward_min)

    # z0 = jnp.clip(z0, a_min=-env.reward_max, a_max=-env.reward_max)

    z_key, key = jax.random.split(key, 2)
    p = 0.3
    rng = jax.random.uniform(z_key, (1, 1))
    z0 = jnp.where(rng > 0.7, -env.reward_max, z0)  # use z min
    z0 = jnp.where(rng < 0.2, -env.reward_min, z0)  # use z max

    z0 = jnp.repeat(z0, env.num_agents, axis=0)

    def body(data, key_):
        graph, rnn_state, z = data
        action, log_pi, new_rnn_state = actor(graph, rnn_state, key_, z=z)
        next_graph, reward, cost, done, info = env.step(graph, action)

        # z dynamics
        z_next = (z + reward) / gamma
        z_next = jnp.clip(z_next, -env.reward_max, -env.reward_min)

        return ((next_graph, new_rnn_state, z_next),
                (graph, action, rnn_state, reward, cost, done, log_pi, next_graph, z))

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs, zs) = (
        jax.lax.scan(body, (init_graph, init_rnn_state, z0), keys, length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs, zs)
    return rollout_data


def test_rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        # init_Vh_rnn_state: Optional[Array] = None,
        z_fn: Optional[Callable] = None,
        stochastic: bool = False,
        use_fixed_reset: bool = False,
):
    key_x0, key = jax.random.split(key)
    if use_fixed_reset:
        init_graph = env.reset_test(key)
    else:
        init_graph = env.reset(key_x0)
    z0 = jax.random.uniform(key, (env.num_agents, 1), minval=-env.reward_max, maxval=-env.reward_min)

    def body_(data, key_):
        graph, rnn_state, i = data
        if z_fn is not None:
            z = z_fn(graph)
        else:
            z = z0
            # new_Vh_rnn_state = Vh_rnn_state
        # z = z.repeat(env.num_agents, axis=0)
        if not stochastic:
            action, rnn_state = actor(graph, rnn_state, z=z)
        else:
            action, rnn_state = actor(graph, rnn_state, key_, z=z)
            
        if use_fixed_reset:
            i = i + 1
            
        next_graph, reward, cost, done, info = env.step(graph, action, iter = i / env.max_episode_steps)
        
        return (next_graph, rnn_state, i), (graph, action, rnn_state, reward, cost, done, None, next_graph, z)

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs, zs) = (
        jax.lax.scan(body_,
                     (init_graph, init_rnn_state, 0),
                     keys,
                     length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs, zs)
    return rollout_data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def empty_grad_tx() -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return None, None

    return optax.GradientTransformation(init_fn, update_fn)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def is_connected():
    try:
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close()
        return True
    except OSError:
        pass
    print('No internet connection')
    return False


def plot_cbf(
        fig: plt.Figure,
        cbf: Callable,
        env: MultiAgentEnv,
        graph: GraphsTuple,
        agent_id: int,
        x_dim: int,
        y_dim: int,
) -> plt.Figure:
    ax = fig.gca()
    n_mesh = 30
    low_lim, high_lim = env.state_lim(graph.states)
    x, y = jnp.meshgrid(
        jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh),
        jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    )
    states = graph.states

    # generate new states
    plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    plot_states = plot_states.at[:, :, agent_id, x_dim].set(x)
    plot_states = plot_states.at[:, :, agent_id, y_dim].set(y)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    new_graph = get_new_graph_vmap(plot_states)
    h = jax.vmap(jax.vmap(cbf))(new_graph)[:, :, agent_id, :].squeeze(-1)
    plt.contourf(x, y, h, cmap=sns.color_palette("rocket", as_cmap=True), levels=15, alpha=0.5)
    plt.colorbar()
    plt.contour(x, y, h, levels=[0.0], colors='blue')
    ax.set_xlim(low_lim[0], high_lim[0])
    ax.set_ylim(low_lim[1], high_lim[1])
    plt.axis('off')

    return fig


def get_bb_cbf(cbf: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int):
    n_mesh = 20
    low_lim = jnp.array([0, 0])
    high_lim = jnp.array([env.area_size, env.area_size])
    b_xs = jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh)
    b_ys = jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
    states = graph.states

    # generate new states
    bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, x_dim].set(bb_Xs)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, y_dim].set(bb_Ys)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    bb_new_graph = get_new_graph_vmap(bb_plot_states)
    bb_h = jax.vmap(jax.vmap(cbf))(bb_new_graph)[:, :, agent_id, :].squeeze(-1)
    assert bb_h.shape == (n_mesh, n_mesh)
    return b_xs, b_ys, bb_h


def get_bb_Vh(Vh: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int):
    n_mesh = 20
    low_lim = jnp.array([0, 0])
    high_lim = jnp.array([env.area_size, env.area_size])
    b_xs = jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh)
    b_ys = jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
    states = graph.states

    # generate new states
    bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, x_dim].set(bb_Xs)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, y_dim].set(bb_Ys)

    # get new graphs
    def get_new_graph_(graph_, states_):
        senders = graph_.senders
        receivers = graph_.receivers
        graph_ = graph_.replace(states=states_, edges=states_[receivers] - states_[senders])
        return graph_

    bb_plot_graphs = jax.vmap(jax.vmap(ft.partial(get_new_graph_, graph)))(bb_plot_states)

    bb_Vh = jax.vmap(jax.vmap(Vh))(bb_plot_graphs)[:, :, agent_id, :].squeeze(-1)

    assert bb_Vh.shape == (n_mesh, n_mesh)
    return b_xs, b_ys, bb_Vh


def centered_norm(vmin, vmax):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)


def plot_rnn_states(rnn_states: Array, name: str, path: str):
    """
    rnn_states: (T, n_layer, n_agent, n_carry, hid_size)
    """
    T, n_layer, n_agent, n_carry, hid_size = rnn_states.shape
    for i_layer in range(n_layer):
        fig, ax = plt.subplots(nrows=n_agent, ncols=n_carry, figsize=(10, 20))
        for i_agent in range(n_agent):
            for i_carry in range(n_carry):
                ax[i_agent, i_carry].plot(rnn_states[:, i_layer, i_agent, i_carry, :])
                ax[i_agent, i_carry].set_title(f'Agent {i_agent}, carry {i_carry}, layer {i_layer}')
                ax[i_agent, i_carry].set_xlabel('Time step')
                ax[i_agent, i_carry].set_ylabel('State value')
        fig.tight_layout()
        plt.savefig(os.path.join(path, f'rnn_states_{name}_layer{i_layer}.png'))
