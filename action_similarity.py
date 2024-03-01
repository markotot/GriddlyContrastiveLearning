import copy
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from griddly.RenderTools import VideoRecorder
from torch.utils.tensorboard import SummaryWriter

from core.config_generator import get_background_env_configs
from core.environment import make_env, make_pcg_env
from core.network import PPONetwork
from utils.videos import record_video
from contrastive_cnn import load_dataset

def action_similarity_on_level(ckpt_path, env_configs, env_level):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_total_actions = 100

    envs = gym.vector.SyncVectorEnv(
            [make_env(f"configs/{env_configs[i]}", 0, 0 ,0 ,0, level=env_level)
                for i in range(len(env_configs))])
    agent = PPONetwork(envs).to(device)

    agent.load_state_dict(torch.load(f"checkpoints/ppo/{ckpt_path}"))
    agent.eval()

    selected_actions = np.zeros(shape=(len(env_configs), len(env_configs)), dtype=np.int32)
    next_obs = torch.Tensor(envs.reset()).to(device)
    for step in range(0, num_total_actions):

        actions, _, _, _, _ = agent.get_action_and_value_and_latent(next_obs, greedy=True)

        for i in range(len(env_configs)):
            for j in range(len(env_configs)):
                if actions[i] == actions[j]:
                    selected_actions[i, j] += 1

        actions = actions.cpu().numpy()
        actions = np.ones_like(actions) * actions[0]

        obs, reward, done, info = envs.step(actions)
        next_obs = torch.Tensor(obs).to(device)

    print(f"Level {env_level} action similarity")
    print(selected_actions)


def action_similarity_on_dataset():
    pass
# main function
if __name__ == "__main__":

    ckpt_path = "weights_['butterflies-floor-wall-boxes-knight.yaml'].ckpt"
    # ckpt_path = "ppo-cnn-983.ckpt"

    # experiment_type = "fully-observable-back"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-2-grass",
    #     f"{experiment_type}/cluster-3-orange",
    #     f"{experiment_type}/cluster-4-lbrown",
    #     f"{experiment_type}/cluster-5-lblue",
    #     f"{experiment_type}/cluster-6-biege",
    #     f"{experiment_type}/cluster-7-space",
    #     f"{experiment_type}/cluster-8-grey",
    #     f"{experiment_type}/cluster-9-red",
    #     f"{experiment_type}/cluster-10-fill",
    # ]

    # experiment_type = "fully-observable-walls"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-1-floor-barrel",
    #     f"{experiment_type}/cluster-1-floor-doors",
    #     f"{experiment_type}/cluster-1-floor-evil-trees",
    #     f"{experiment_type}/cluster-1-floor-fence",
    #     f"{experiment_type}/cluster-1-floor-fire",
    #     f"{experiment_type}/cluster-1-floor-mineral",
    #     f"{experiment_type}/cluster-1-floor-number",
    #     f"{experiment_type}/cluster-1-floor-pipe",
    #     f"{experiment_type}/cluster-1-floor-trees",
    # ]

    # experiment_type = "fully-observable-walls"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-1-floor-barrel",
    #     f"{experiment_type}/cluster-1-floor-doors",
    #     f"{experiment_type}/cluster-1-floor-evil-trees",
    #     f"{experiment_type}/cluster-1-floor-fence",
    #     f"{experiment_type}/cluster-1-floor-fire",
    #     f"{experiment_type}/cluster-1-floor-mineral",
    #     f"{experiment_type}/cluster-1-floor-number",
    #     f"{experiment_type}/cluster-1-floor-pipe",
    #     f"{experiment_type}/cluster-1-floor-trees",
    # ]

    game_name = "butterflies"
    env_configs, _ = get_background_env_configs(template_name=game_name, train=False)

    action_similarity_on_level(ckpt_path=ckpt_path, env_configs=env_configs, env_level=0)
    action_similarity_on_level(ckpt_path=ckpt_path, env_configs=env_configs, env_level=1)

