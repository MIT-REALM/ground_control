import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml

from matplotlib import pyplot as plt

from cmarl.algo import make_algo, EFInforMARL
from cmarl.env import make_env
from cmarl.env.base import RolloutResult
from cmarl.trainer.data import Rollout
from cmarl.trainer.utils import get_bb_cbf, plot_rnn_states, test_rollout, get_bb_Vh
from cmarl.utils.graph import GraphsTuple
from cmarl.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap, np2jax, jax2np
from cmarl.utils.plot import get_bb_Vh


def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
    if hasattr(config, "delta_scale"):
        delta_scale = config.delta_scale
    else:
        delta_scale = 10.0 if args.delta_scale is None else args.delta_scale
    # create environments
    if hasattr(config, "goal_reward_scale"):
        goal_reward_scale = config.goal_reward_scale
    else:
        goal_reward_scale = 1.0
        
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    if args.use_fixed_reset:
        num_agents = 2
        args.n_mov_obs = 4
        args.obs = 1
        args.epi = 1
    
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        max_travel=args.max_travel,
        full_observation=args.full_observation,
        n_mov_obs=args.n_mov_obs,
        delta_scale=delta_scale,
        goal_reward_scale=goal_reward_scale,
    )

    if args.path is not None:
        path = args.path
        model_path = os.path.join(path, "models")
        if args.step is None:
            models = os.listdir(model_path)
            step = max([int(model) for model in models if model.isdigit()])
        else:
            step = args.step
        print("step: ", step)

        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            cost_weight=config.cost_weight,
            actor_gnn_layers=config.actor_gnn_layers,
            Vl_gnn_layers=config.Vl_gnn_layers,
            Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
            lr_actor=config.lr_actor,
            lr_Vl=config.lr_Vl,
            max_grad_norm=2.0,
            seed=config.seed,
            use_rnn=config.use_rnn,
            rnn_layers=config.rnn_layers,
            use_lstm=config.use_lstm,
        )
        algo.load(model_path, step)
        if args.stochastic:
            def act_fn(x, z, rnn_state, key):
                action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
                return action, new_rnn_state
            act_fn = jax.jit(act_fn)
        else:
            act_fn = algo.act
        z_fn = algo.get_opt_z if hasattr(algo, "get_opt_z") else None
        if args.z is not None:
            if args.z == "min":
                z_fn = lambda graph, value_rnn_state: \
                    (jnp.array([[-env.reward_max]]).repeat(env.num_agents, axis=0), value_rnn_state)
            elif args.z == "max":
                z_fn = lambda graph, value_rnn_state: \
                    (jnp.array([[-env.reward_min]]).repeat(env.num_agents, axis=0), value_rnn_state)
            else:
                raise ValueError(f"Unknown z: {args.z}")
        # act_fn = lambda x, z, rnn_state, key: algo.act(x, z, rnn_state)
        act_fn = jax.jit(act_fn)
        init_rnn_state = algo.init_rnn_state
        # if hasattr(algo, "init_Vh_rnn_state"):
        #     init_Vh_rnn_state = algo.init_Vh_rnn_state
        # else:
        #     init_Vh_rnn_state = None
        # init_value_rnn_state = algo.init_value_rnn_state
    else:
        algo = make_algo(
            algo=args.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            alpha=args.alpha,
        )
        act_fn = jax.jit(algo.act)
        path = os.path.join(f"./logs/{args.env}/{args.algo}")
        if not os.path.exists(path):
            os.makedirs(path)
        step = None
        init_rnn_state = None
        # init_value_rnn_state = None
        z_fn = None

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    if args.nojit_rollout:
        print("Only jit step, no jit rollout!")
        rollout_fn = env.rollout_fn_jitstep(act_fn, args.max_step, noedge=True, nograph=args.no_video)
        is_unsafe_fn = None
        reach_rate_fn = None
    else:
        print("jit rollout!")
        rollout_fn = ft.partial(test_rollout,
                                env,
                                act_fn,
                                init_rnn_state,
                                z_fn=z_fn,
                                stochastic=args.stochastic,
                                use_fixed_reset=args.use_fixed_reset)
        rollout_fn = jax_jit_np(rollout_fn)
        is_unsafe_fn = jax_jit_np(jax_vmap(env.unsafe_mask))
        is_slow_mov_obs_coll_fn = jax_jit_np(jax_vmap(env.slow_mov_obs_collision_mask))
        is_fast_mov_obs_coll_fn = jax_jit_np(jax_vmap(env.fast_mov_obs_collision_mask))
        is_reach_fn = jax_jit_np(jax_vmap(env.goal_mask))

    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_slow_mov_obs_colls = []
    is_fast_mov_obs_colls = []
    is_reaches = []
    rates = []

    bTah_Vh = []
    bTa_Vl = []

    if args.vh_contour is not None:
        assert isinstance(algo, EFInforMARL)
        def Vh_fn_(graph, z):
            pass
        get_bb_Vh_fn_ = ft.partial(get_bb_Vh, algo.get_value, env, agent_id=args.vh_contour, x_dim=0, y_dim=1)


        # cbf_fn = jax.jit(algo.get_cbf)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h


    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        if args.nojit_rollout:
            raise NotImplementedError
        else:
            rollout = rollout_fn(key_x0)
            is_unsafes.append(is_unsafe_fn(rollout.graph))
            is_reaches.append(is_reach_fn(rollout.graph))
            is_slow_mov_obs_colls.append(is_slow_mov_obs_coll_fn(rollout.graph))
            is_fast_mov_obs_colls.append(is_fast_mov_obs_coll_fn(rollout.graph))
            

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        reach_rate = is_reaches[-1].max(axis=0).mean()
        slow_mov_obs_coll_rate = is_slow_mov_obs_colls[-1].max(axis=0).max(axis=-1).mean()
        fast_mov_obs_coll_rate = is_fast_mov_obs_colls[-1].max(axis=0).max(axis=-1).mean()
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate * 100:.3f}%, reach rate: {reach_rate * 100:.3f}%, slow_mov_obs_coll rate: {slow_mov_obs_coll_rate * 100:.3f}%, fast_mov_obs_coll rate: {fast_mov_obs_coll_rate * 100:.3f}%")

        rates.append(np.array(safe_rate))

        if args.plot_vh:
            Tah_Vh = jax.vmap(algo.get_Vh)(rollout.graph, rollout.rnn_states)
            # Tah_Vh = jax.vmap(ft.partial(algo.get_Vh, params={'Vh': algo.Vh_train_state.params}))(rollout.graph, rollout.zs, rollout.rnn_states)
            # (Ta_Vl, Tah_Vh), _, _ = algo.scan_value(rollout, init_Vh_rnn_state[:, 0, :, :][:, None, :, :], init_Vh_rnn_state, algo.critic_train_state.params, algo.Vh_train_state.params)
            bTah_Vh.append(Tah_Vh)
            # bTa_Vl.append(Ta_Vl)
    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    is_slow_mov_obs_coll = np.mean(np.max(np.max(np.stack(is_slow_mov_obs_colls), axis=1), axis=-1), axis=-1)
    is_fast_mov_obs_coll = np.mean(np.max(np.max(np.stack(is_fast_mov_obs_colls), axis=1), axis=-1), axis=-1)
    is_reach = np.max(np.stack(is_reaches), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    reach_mean, reach_std = is_reach.mean(), is_reach.std()
    slow_mov_obs_col_mean, mov_obs_col_std = is_slow_mov_obs_coll.mean(), is_slow_mov_obs_coll.std()
    fast_mov_obs_col_mean, mov_obs_col_std = is_fast_mov_obs_coll.mean(), is_fast_mov_obs_coll.std()
    # breakpoint()
    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, reach_rate: {reach_mean * 100:.3f}%, " 
        f"mov_coll_rate: {slow_mov_obs_col_mean * 100:.3f}% ,"
        f"fast_mov_coll_rate: {fast_mov_obs_col_mean * 100:.3f}%"
    )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f},"
                    f"{reach_mean * 100:.3f},{reach_std * 100:.3f},"
                    f"{slow_mov_obs_col_mean * 100:.3f},{mov_obs_col_std * 100:.3f},"
                    f"{fast_mov_obs_col_mean * 100:.3f},{mov_obs_col_std * 100:.3f},")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        viz_opts = {}
        if args.plot_vh:
            viz_opts["Vh"] = bTah_Vh[ii]
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--plot-rnn", action="store_true", default=False)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("-z", type=str, default=None)
    parser.add_argument("--plot-vh", action="store_true", default=False)
    parser.add_argument('--vh-contour',type=int, default=None)
    parser.add_argument("--full-observation", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--n_mov_obs", type=int, default=4)
    parser.add_argument("--use_fixed_reset", action="store_true", default=False)
    parser.add_argument("--delta_scale", type=float, default=None)


    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
