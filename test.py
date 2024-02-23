import core.runner_lstm
import core.runner_ppo

if __name__ == "__main__":

    model = "ppo"

    experiment_type = "fully-observable-back"
    env_configs = [
        f"{experiment_type}/cluster-1-floor",
        f"{experiment_type}/cluster-2-grass",
        f"{experiment_type}/cluster-3-orange",
        f"{experiment_type}/cluster-4-lbrown",
        f"{experiment_type}/cluster-5-lblue",
        f"{experiment_type}/cluster-6-biege",
        f"{experiment_type}/cluster-7-space",
        f"{experiment_type}/cluster-8-grey",
        f"{experiment_type}/cluster-9-red",
        f"{experiment_type}/cluster-10-fill",
    ]

    # experiment_type = "fully-observable-player"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-1-floor-alien",
    #     f"{experiment_type}/cluster-1-floor-angel",
    #     f"{experiment_type}/cluster-1-floor-coins",
    #     f"{experiment_type}/cluster-1-floor-necromancer",
    #     f"{experiment_type}/cluster-1-floor-rogue",
    #     f"{experiment_type}/cluster-1-floor-wolf",
    # ]

    # experiment_type = "fully-observable-walls"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-1-floor-doors",
    #     f"{experiment_type}/cluster-1-floor-fence",
    #     f"{experiment_type}/cluster-1-floor-fire",
    #     f"{experiment_type}/cluster-1-floor-trees",
    # ]

    # experiment_type = "fully-observable-boxes"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-1-floor-armor",
    #     f"{experiment_type}/cluster-1-floor-cars",
    #     f"{experiment_type}/cluster-1-floor-boxes2",
    #     f"{experiment_type}/cluster-1-floor-chess",
    #     f"{experiment_type}/cluster-1-floor-food",
    # ]

    # experiment_type = "fully-observable-mix"
    # env_configs = [
    #     f"{experiment_type}/cluster-1-floor",
    #     f"{experiment_type}/cluster-2-grass-doors-alien-cars",
    #     f"{experiment_type}/cluster-4-lbrown-trees-angel-boxes2",
    #     f"{experiment_type}/cluster-5-lblue-fence-rogue-armor",
    #     f"{experiment_type}/cluster-9-red-fire-coins-chess",
    #     f"{experiment_type}/cluster-10-fill-fence-necromancer-food",
    # ]




    ckpt_path = "weights_['cluster-1-floor'].ckpt"

    pcg = False

    if model == "ppo":
        core.runner_ppo.test(env_names=env_configs, ckpt_path=ckpt_path, pcg=pcg, total_num_episodes=50)
    elif model == "lstm":
        core.runner_lstm.test(env_names=env_configs, ckpt_path=ckpt_path, pcg=pcg, total_num_episodes=50)

