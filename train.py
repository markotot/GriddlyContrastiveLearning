import argparse
import os
from distutils.util import strtobool

import core.runner_ppo
import core.runner_lstm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ClusterGriddly",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="GriddlyCluster",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-video-episodes", type=int, default=5,
        help="the number of episodes to save as videos")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


if __name__ == "__main__":
    args = parse_args()

    experiment_type = "fully-observable-walls"
    # experiment_type = "partially-observable"
    env_configs = [
        f"{experiment_type}/cluster-1-floor",
        f"{experiment_type}/cluster-1-floor-doors",
        f"{experiment_type}/cluster-1-floor-fence",
        f"{experiment_type}/cluster-1-floor-trees",
        f"{experiment_type}/cluster-1-floor-food",

        # f"{experiment_type}/cluster-1-floor-necromancer",
        # f"{experiment_type}/cluster-2-grass",
        # f"{experiment_type}/cluster-3-orange",
        # f"{experiment_type}/cluster-4-lbrown",
        # f"{experiment_type}/cluster-5-lblue",
        # f"{experiment_type}/cluster-6-biege",
        # f"{experiment_type}/cluster-7-space",
        # f"{experiment_type}/cluster-8-grey",
        # f"{experiment_type}/cluster-9-red",
        # f"{experiment_type}/cluster-10-fill",
    ]
    model = "ppo"
    total_steps = 1_000_000
    # load_ckpt_path = "clusters-1-8-ppo-2mil.ckpt"
    load_ckpt_path = "weights_['cluster-1-floor', 'cluster-1-floor-doors', 'cluster-1-floor-fence', 'cluster-1-floor-trees'].ckpt"
    freeze_cnn = False
    pcg = False

    if model == "ppo":
        core.runner_ppo.run(args, env_configs, pcg=pcg, total_steps=total_steps, ckpt_path=load_ckpt_path, freeze_cnn=freeze_cnn)
    elif model == "lstm":
        core.runner_lstm.run(args, env_configs, pcg=pcg, total_steps=total_steps, ckpt_path=load_ckpt_path, freeze_cnn=freeze_cnn)


