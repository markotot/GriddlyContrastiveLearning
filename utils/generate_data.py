import copy
import sys
from datetime import time
from time import sleep
import matplotlib.pyplot as plt
import torch

import numpy as np

from core.environment import make_env
from core.level_generator import generate_levels
from core.network import PPONetwork


def get_random_actions(num_total_actions, env_config):

    env = make_env(f"configs/{env_config}", 0, 0, 0, 0)()
    env.reset()

    actions = np.zeros(shape=(num_total_actions,))
    for step in range(0, num_total_actions):
        a = np.random.randint(0, env.action_space.n)
        # a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        if done:

            env.reset()
        actions[step] = a

    return actions


def get_agent_actions(num_total_actions, env_config, ckpt_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(f"configs/{env_config}", 0, 0, 0, 0)()
    env.single_action_space = env.action_space

    agent = PPONetwork(env).to(device)
    agent.load_checkpoint(ckpt_path)
    actions = np.zeros(shape=(num_total_actions,))
    obs = env.reset()
    obs = torch.tensor(obs).to(device)

    for step in range(0, num_total_actions):

        action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
        obs, reward, done, info = env.step(action.cpu().numpy())
        obs = torch.from_numpy(obs).to(device)

        if done:
            env.reset()
        actions[step] = action

    return actions


def collect_observations(env_config, actions, level_strings):

    env = make_env(f"configs/{env_config}", 0, 0, 0, 0)()

    if level_strings is not None:
        observation = env.reset(level_string=level_strings[0])
    else:
        observation = env.reset()

    all_observations = np.zeros(shape=(len(actions), *observation.shape), dtype=np.uint8)

    num_episode = 1
    for step, a in enumerate(actions):

        obs, reward, done, info = env.step(a)
        if done:
            if level_strings is not None:
                env.reset(level_string=level_strings[num_episode])
            else:
                env.reset()
            num_episode += 1
        else:
            all_observations[step] = obs

    return all_observations


if __name__ == "__main__":

    np.random.seed(0)

    num_random_actions = 120_000
    # num_agent_actions = 1

    level_strings = generate_levels(2000, 0, 7, 7)

    random_actions = get_random_actions(num_random_actions,
                                        env_config="fully-observable-back/cluster-1-floor.yaml")

    # agent_actions = get_agent_actions(num_agent_actions,
    #                                   env_config="cluster-1-floor.yaml",
    #                                   ckpt_path="weights_['cluster-1'].ckpt")
    print("Actions collected")

    experiment_name = "fully-observable-back"
    env_configs = [
        f"{experiment_name}/cluster-1-floor.yaml",
        f"{experiment_name}/cluster-2-grass.yaml",
        f"{experiment_name}/cluster-3-orange.yaml",
        f"{experiment_name}/cluster-4-lbrown.yaml",
        f"{experiment_name}/cluster-5-lblue.yaml",
        f"{experiment_name}/cluster-6-biege.yaml",
        f"{experiment_name}/cluster-7-space.yaml",
        f"{experiment_name}/cluster-8-grey.yaml",
        f"{experiment_name}/cluster-9-red.yaml",
        f"{experiment_name}/cluster-10-fill.yaml",
    ]

    print(f"Collecting random observations")
    all_observations = {}
    for env_config in env_configs:
        all_observations[env_config] = collect_observations(env_config, random_actions, level_strings=level_strings)

    # print(f"Collecting agent observations")
    # for env_config in env_configs:
    #     agent_observations = collect_observations(env_config, agent_actions, level_strings=None)
    #     all_observations[env_config] = np.concatenate((all_observations[env_config], agent_observations))

    for key in all_observations.keys():
        obs = np.moveaxis(all_observations[key][0], 0, -1)
        plt.imshow(obs)
        plt.show()

    print(f"Filtering observations")
    filtered_observations = {}
    _, idx = np.unique(all_observations[env_configs[0]], axis=0, return_index=True)
    for env_config in env_configs:

        filtered_observations[env_config] = all_observations[env_config][np.sort(idx)]
        print(f"Filtered observations for {env_config}, size {len(filtered_observations[env_config])}")

    print(len(filtered_observations[env_config]))

    for env_config in env_configs:
        np.savez(f"datasets/{env_config}.npz", filtered_observations[env_config])



