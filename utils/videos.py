import copy
import random
import time

import gym
import numpy as np
import torch

from griddly.RenderTools import VideoRecorder


from core.environment import make_env

def record_video(agent, env_names, total_num_episodes, global_step):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_recorder = VideoRecorder()

    for env_name in env_names:

        env = make_env(f"configs/{env_name}.yaml", 0, 0,0,0)()
        env.single_action_space = env.action_space
        obs = env.reset()

        video_path = f"videos/training/{env_name.split('/')[-1].split('.')[0]}-{global_step}.mp4"
        video_frame = np.moveaxis(obs, 0, -1)
        video_recorder.start(video_path, video_frame.shape)

        obs = torch.tensor(obs).to(device)
        num_episodes = 0
        while num_episodes < total_num_episodes:

            action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            obs, reward, done, info = env.step(action.cpu().numpy())
            video_frame = np.rot90(np.moveaxis(obs, 0, -1), 3)
            video_recorder.add_frame(video_frame)
            if done:
                num_episodes += 1
                obs = env.reset()
            obs = torch.from_numpy(obs).to(device)

        video_recorder.close()
        print(f"Video for {env_name}-{global_step} recoderd successfully")
