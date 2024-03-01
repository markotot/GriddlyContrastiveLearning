import time
import torch
import core.runner_lstm
import core.runner_ppo
from core.environment import make_pcg_env, make_env
from core.network import PPONetwork
import torch
import torch.nn.functional as F
def run_game():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = "ppo"
    experiment_type = "generated"

    # env_config = "butterflies-biege-wall-boxes-knight.yaml"
    env_config = "dwarf-floor-wall-boxes-knight.yaml"
    # env_config = "cluster-5-lblue-fence-rogue-armor.yaml"
    # env_config = "cluster-9-red-fire-coins-chess.yaml"

    # ckpt_path = "weights_['cluster-1-floor'].ckpt"
    pcg = False
    env_name = f"{experiment_type}/{env_config}"
    actions_sequence = []
    if pcg:
        env = make_pcg_env(f"configs/{env_name}", 0, 0, 0, 0)()
    else:
        env = make_env(f"configs/{env_name}", 0, 0, 0, 0)()

    env.single_action_space = env.action_space
    agent = PPONetwork(env).to(device)
    # agent.load_checkpoint(ckpt_path)

    actions = [3, 4, 1, 2, 2, 3, 4, 4,
               1, 4, 4, 1, 1, 3, 2, 2, 2, 1,
               1, 4, 4, 2, 1, 4, 4, 3, 3,
               3, 2, 3, 4
               ]

    rewards = 0
    while True:
        obs = env.reset()
        obs = torch.tensor(obs).to(device)
        for action, idx in enumerate(actions):
            env.render()

            a, probs, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            print(a)
            obs, reward, done, info = env.step(a.cpu().numpy())
            # obs, reward, done, info = env.step(actions[idx])
            obs = torch.tensor(obs).to(device)
            rewards += reward
            time.sleep(1)
            if done:
                print(rewards)
                break
        env.close()

if __name__ == "__main__":
    #run_game()

    temperature = 0.1
    logits = torch.tensor([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

    loss = F.cross_entropy(logits, labels, reduction='mean')
    print(loss)
