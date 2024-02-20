import time

import core.runner_lstm
import core.runner_ppo
from core.environment import make_pcg_env, make_env

if __name__ == "__main__":

    model = "ppo"

    experiment_type = "fully-observable-mix"

    env_config = "cluster-2-grass-doors-alien-cars.yaml"
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

    actions = [3, 4, 1, 2, 2, 3, 4, 4,
               1, 4, 4, 1, 1, 3, 2, 2, 2, 1,
                1, 4, 4, 2, 1, 4, 4, 3, 3,
                3, 2, 3, 4
               ]

    rewards = 0
    while True:
        env.reset()
        for action in actions:
            env.render()
            obs, reward, done, info = env.step(action)
            rewards += reward
            time.sleep(0.1)
            if done:
                print(rewards)
                break
        env.close()