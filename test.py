import core.runner_lstm
import core.runner_ppo
from core.config_generator import get_mixed_env_configs, get_background_env_configs

if __name__ == "__main__":

    model = "ppo"

    game_name = "dwarf"

    env_configs, _ = get_background_env_configs(template_name=game_name, train=False)
    # env_configs, _ = get_mixed_env_configs(template_name=game_name, train=False)
    env_level = 0
    ckpt_path = "weights_dwarf_group.ckpt"
    # ckpt_path = "ppo-cnn-983.ckpt"

    pcg = False

    if model == "ppo":
        core.runner_ppo.test(env_names=env_configs, ckpt_path=ckpt_path, pcg=pcg, total_num_episodes=50, env_level=env_level)
    elif model == "lstm":
        core.runner_lstm.test(env_names=env_configs, ckpt_path=ckpt_path, pcg=pcg, total_num_episodes=50)

