import gym
from griddly import GymWrapperFactory
from core.level_generator import generate_levels

### IMPORTANT:
# Research/site-packages/griddly/GymWrapper.py was modified!
# Class PCGGymWrapper was added to the file.
# Method wrapper.build_pcg_gym_from_yaml() was added to the file

def make_env(env_config, seed, idx, capture_video, run_name, level=0):
    def thunk():

        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml('MyEnv', env_config, level=level, max_steps=50)

        env = gym.make('GDY-MyEnv-v0')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.observation_space = gym.spaces.box.Box(low=0, high=255, shape=(3, 84, 84))

        return env

    return thunk

def make_pcg_env(env_config, seed, idx, capture_video, run_name, level=0):
    def thunk():

        wrapper = GymWrapperFactory()

        level_list = generate_levels(1, seed=seed, width=12, height=12)
        wrapper.build_pcg_gym_from_yaml('MyEnv',
                                        env_config,
                                        level=level,
                                        max_steps=512,
                                        level_list=level_list)

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