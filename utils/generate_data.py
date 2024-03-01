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
from pathlib import Path

from core.config_generator import get_background_env_configs, get_mixed_env_configs

def get_random_actions(num_total_actions, env_config):

    env = make_env(f"../configs/{env_config}", 0, 0, 0, 0)()
    actions = np.random.randint(0, env.action_space.n, size=(num_total_actions,))

    return actions


def get_agent_actions(num_total_actions, env_config, ckpt_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(f"../configs/{env_config}", 0, 0, 0, 0)()
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

    env = make_env(f"../configs/{env_config}", 0, 0, 0, 0)()

    if level_strings is not None:
        observation = env.reset(level_string=level_strings[0])
    else:
        observation = env.reset()

    all_observations = np.zeros(shape=(len(actions), *observation.shape), dtype=np.uint8)

    num_episode = 1
    steps_per_episode = 0
    for step, a in enumerate(actions):

        obs, reward, done, info = env.step(a)
        steps_per_episode += 1
        if done or steps_per_episode >= 100:
            steps_per_episode = 0
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

    num_random_actions = 200_000

    game_name = "dwarf"
    env_configs, _ = get_background_env_configs(template_name=game_name, train=False)

    level_strings = generate_levels(game_name, 50000, 0, 7, 7)

    random_actions = get_random_actions(num_random_actions,
                                        env_config=f"{game_name}.yaml")

    print("Actions collected")


    for env_config in env_configs:
        print(f"Collecting random observations")
        all_observations = {}
        all_observations[env_config] = collect_observations(env_config, random_actions, level_strings=level_strings)

        for key in all_observations.keys():
            obs = np.moveaxis(all_observations[key][0], 0, -1)
            plt.imshow(obs)
            plt.show()

        print(f"Filtering observations")
        file_path = f"../datasets/{game_name}_unique_idx.npz"
        my_file = Path(file_path)
        if my_file.is_file():
            unique_idx = np.load(file_path)
            idx = unique_idx["idx"]
        else:
            _, idx = np.unique(all_observations[env_configs[0]], axis=0, return_index=True)
            np.savez(file_path, idx=idx)

        filtered_observations = all_observations[env_config][np.sort(idx)]
        np.savez(f"../datasets/{env_config.split('/')[-1]}.npz", filtered_observations)
        print(f"Filtered observations for {env_config}, size {len(filtered_observations)}")


