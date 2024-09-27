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

from ..env.mpe import MPE, MPEEnvState
from ..trainer.data import Rollout

if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


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
