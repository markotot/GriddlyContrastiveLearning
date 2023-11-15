import core.runner_lstm
import core.runner_ppo

if __name__ == "__main__":

    model = "ppo"
    env_config = "cluster-9-red-walls-spikes.yaml"
    # env_config = "cluster-1-floor-walls.yaml"
    ckpt_path = "ppo_1_after_c-cnn-6.ckpt"

    if model == "ppo":
        core.runner_ppo.test(env_name=env_config, ckpt_path=ckpt_path, total_num_episodes=20)
    elif model == "lstm":
        core.runner_lstm.test(env_name=env_config, ckpt_path=ckpt_path, total_num_episodes=20)

