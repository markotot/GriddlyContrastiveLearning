import gym
from griddly import GymWrapperFactory


def make_env(env_config, seed, idx, capture_video, run_name, level=0):
    def thunk():

        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml('MyEnv', env_config, level=level, max_steps=200)

        env = gym.make('GDY-MyEnv-v0')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.observation_space = gym.spaces.box.Box(low=0, high=255, shape=(3, 84, 84))

        return env

    return thunk