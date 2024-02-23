import time
import torch
import core.runner_lstm
import core.runner_ppo
from core.environment import make_pcg_env, make_env
from core.network import PPONetwork

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = "ppo"
    experiment_type = "fully-observable-mix"
    env_config = "cluster-1-floor.yaml"
    # env_config = "cluster-4-lbrown-trees-angel-boxes2.yaml"
    # env_config = "cluster-5-lblue-fence-rogue-armor.yaml"
    # env_config = "cluster-9-red-fire-coins-chess.yaml"

    ckpt_path = "weights_['cluster-1-floor'].ckpt"
    pcg = False
    env_name = f"{experiment_type}/{env_config}"
    actions_sequence = []
    if pcg:
        env = make_pcg_env(f"configs/{env_name}", 0, 0, 0, 0)()
    else:
        env = make_env(f"configs/{env_name}", 0, 0, 0, 0)()

    env.single_action_space = env.action_space
    agent = PPONetwork(env).to(device)
    agent.load_checkpoint(ckpt_path)

    actions = [3, 4, 1, 2, 2, 3, 4, 4,
               1, 4, 4, 1, 1, 3, 2, 2, 2, 1,
                1, 4, 4, 2, 1, 4, 4, 3, 3,
                3, 2, 3, 4
               ]

    rewards = 0
    while True:
        obs = env.reset()
        obs = torch.tensor(obs).to(device)
        for action in actions:
            env.render()

            a, probs, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            print(a)
            obs, reward, done, info = env.step(a.cpu().numpy())
            obs = torch.tensor(obs).to(device)
            rewards += reward
            time.sleep(1)
            if done:
                print(rewards)
                break
        env.close()