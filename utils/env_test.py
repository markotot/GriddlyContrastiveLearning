from time import sleep

from core.level_generator import generate_levels
from core.environment import make_env
import numpy as np

level_strings = generate_levels(1)


env_name = "cluster-9-red-walls-spikes.yaml"
env = make_env(f"../configs/{env_name}", 0, 0, 0, 0)()
env.reset(level_string=level_strings[0])
num_total_actions = 1000

for step in range(0, num_total_actions):
    # a = np.random.randint(0, env.action_space.n)
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)
    env.render(observer='global')
    if done:
        env.reset(level_string=level_strings[0])