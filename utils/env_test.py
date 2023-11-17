from time import sleep

from core.environment import make_env
import numpy as np

env_name = "cluster-9-red-walls-spikes.yaml"
env = make_env(f"configs/{env_name}", 0, 0, 0, 0)()
env.reset()
num_total_actions = 1000

for step in range(0, num_total_actions):
    # a = np.random.randint(0, env.action_space.n)
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)
    sleep(0.1)
    env.render()
    if done:
        env.reset()