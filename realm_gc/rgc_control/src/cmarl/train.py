import argparse
import datetime
import os
import ipdb
import numpy as np
import wandb
import yaml
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from cmarl.algo import make_algo
from cmarl.env import make_env
from cmarl.trainer.trainer import Trainer
from cmarl.trainer.utils import is_connected


def train(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        # os.environ["JAX_DISABLE_JIT"] = "True"

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        n_mov_obs=args.n_mov_obs,
        max_step=128,
        delta_scale=args.delta_scale,
        goal_reward_scale=args.goal_reward_scale,
    )
    
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        n_mov_obs=args.n_mov_obs,
        max_step=256,
        delta_scale=args.delta_scale,
        goal_reward_scale=args.goal_reward_scale,
    )

    # create algorithm
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=args.cost_weight,
        cbf_weight=args.cbf_weight,
        actor_gnn_layers=args.actor_gnn_layers,
        Vl_gnn_layers=args.Vl_gnn_layers,
        Vh_gnn_layers=args.Vh_gnn_layers,
        rnn_layers=args.rnn_layers,
        lr_actor=args.lr_actor,
        lr_Vl=args.lr_Vl,
        lr_Vh=args.lr_Vh,
        max_grad_norm=2.0,
        alpha=args.alpha,
        cbf_eps=args.cbf_eps,
        seed=args.seed,
        batch_size=args.n_env_train * env._max_step,
        use_rnn=not args.no_rnn,
        use_lstm=args.use_lstm,
        coef_ent=args.coef_ent,
        rnn_step=args.rnn_step,
        gamma=0.99,
        coef_ent_schedule=args.coef_ent_schedule,
        clip_eps=args.clip_eps,
        lagr_init=args.lagr_init,
        lr_lagr=args.lr_lagr,
        use_prev_init=args.use_prev_init,
        use_hl_gauss=args.use_hl_gauss,
        Vh_use_hl_gauss=args.Vh_use_hl_gauss,
        hl_gauss_bins=args.hl_gauss_bins,
        hl_gauss_smooth_ratio=args.hl_gauss_smooth_ratio,
        lr_schedule=args.lr_schedule,
        train_steps=args.steps,
        gamma_schedule=args.gamma_schedule,
        cbf_schedule=args.cbf_schedule,
        alpha_schedule=args.alpha_schedule,
    )
    if args.prev_trained_model is not None:
        try: 
            path = args.prev_trained_model
            with open(os.path.join(path, "config.yaml"), "r") as f:
                config = yaml.load(f, Loader=yaml.UnsafeLoader)
            model_path = os.path.join(path, "models")
            # if args.step is None:
            models = os.listdir(model_path)
            step = max([int(model) for model in models if model.isdigit()])
            # else:
            #     step = args.step
            print("step: ", step)
            if hasattr(config, "cbf_weight"):
                cbf_weight = config.cbf_weight * 4
            else:
                cbf_weight = 1.0
            algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                actor_gnn_layers=config.actor_gnn_layers,
                Vl_gnn_layers=config.Vl_gnn_layers,
                Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
                use_rnn=config.use_rnn,
                rnn_layers=config.rnn_layers,
                use_lstm=config.use_lstm,
                cost_weight=args.cost_weight,
                cbf_weight=cbf_weight,
                lr_actor=args.lr_actor,
                lr_Vl=args.lr_Vl,
                lr_Vh=args.lr_Vh,
                max_grad_norm=2.0,
                alpha=args.alpha,
                cbf_eps=args.cbf_eps,
                seed=args.seed,
                batch_size=args.n_env_train * env._max_step,
                coef_ent=args.coef_ent,
                rnn_step=args.rnn_step,
                gamma=0.99,
                coef_ent_schedule=args.coef_ent_schedule,
                clip_eps=args.clip_eps,
                lagr_init=args.lagr_init,
                lr_lagr=args.lr_lagr,
                use_prev_init=args.use_prev_init,
                use_hl_gauss=args.use_hl_gauss,
                Vh_use_hl_gauss=args.Vh_use_hl_gauss,
                hl_gauss_bins=args.hl_gauss_bins,
                hl_gauss_smooth_ratio=args.hl_gauss_smooth_ratio,
                lr_schedule=args.lr_schedule,
                train_steps=args.steps,
                gamma_schedule=args.gamma_schedule,
                cbf_schedule=args.cbf_schedule,
                alpha_schedule=args.alpha_schedule,
            )
            algo.load(model_path, step)
            print('Loaded model from', model_path)
            # breakpoint()
        except Exception as e:
            print(f"Failed to load due to {e}")
    # set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m%d%H%M%S")
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)
    # if not os.path.exists(f"{args.log_dir}/{args.env}"):
    #     os.makedirs(f"{args.log_dir}/{args.env}")
    if not os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}"):
        os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}", exist_ok=True)
    start_time = int(start_time)
    while os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"):
        start_time += 1
    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}"
    run_name = f"{args.algo}_{start_time}"
    if args.name is not None:
        run_name = run_name + "_" + args.name

    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
        "full_eval_interval": args.full_eval_interval,
    }

    # create trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=0.99,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
    )

    # save config
    wandb.config.update(args)
    wandb.config.update(algo.config, allow_val_change=True)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)

    # start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    # custom arguments
    parser.add_argument("-n", "--num-agents", type=int, default=4)
    parser.add_argument("--algo", type=str, default="gcbfcrpo")
    parser.add_argument("--env", type=str, default="LidarF1TenthTarget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--cost-weight", type=float, default=0.)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument('--full-observation', action='store_true', default=False)
    parser.add_argument("--coef-ent-schedule", action="store_true", default=False)
    parser.add_argument('--clip-eps', type=float, default=0.25)
    parser.add_argument('--lagr-init', type=float, default=0.5)
    parser.add_argument('--lr-lagr', type=float, default=1e-7)
    parser.add_argument("--cbf-weight", type=float, default=1.0)
    parser.add_argument("--cbf-eps", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--lr-schedule", action="store_true", default=False)
    parser.add_argument("--use-prev-init", action="store_true", default=False)
    parser.add_argument("--use-hl-gauss", action="store_true", default=False)
    parser.add_argument("--Vh-use-hl-gauss", action="store_true", default=False)
    parser.add_argument("--hl-gauss-bins", type=int, default=64)
    parser.add_argument("--hl-gauss-smooth-ratio", type=float, default=0.75)
    parser.add_argument("--gamma-schedule", action="store_true", default=False)
    parser.add_argument("--cbf-schedule", action="store_true", default=False)
    parser.add_argument("--alpha-schedule", action="store_true", default=False)

    # arguments
    parser.add_argument("--actor-gnn-layers", type=int, default=1)
    parser.add_argument("--Vl-gnn-layers", type=int, default=1)
    parser.add_argument("--Vh-gnn-layers", type=int, default=1)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-Vl", type=float, default=1e-3)
    parser.add_argument("--lr-Vh", type=float, default=1e-3)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--no-rnn", action="store_true", default=False)
    parser.add_argument("--coef-ent", type=float, default=1e-2)
    parser.add_argument("--rnn-step", type=int, default=16)

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-env-test", type=int, default=32)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--full-eval-interval", type=int, default=100)
    parser.add_argument("--eval-epi", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--n_mov_obs", type=int, default=4)
    parser.add_argument("--prev_trained_model", type=str, default=None)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--goal_reward_scale", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
