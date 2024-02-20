import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from core.environment import make_env, make_pcg_env
from core.network import LSTMPPONetwork

from griddly.RenderTools import VideoRecorder


def run(args, env_config, pcg, total_steps, ckpt_path, freeze_cnn):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #  Having different backgrounds helps it to train faster?
    # env setup

    if pcg:
        env_fn = make_pcg_env
    else:
        env_fn = make_env

    envs = gym.vector.SyncVectorEnv(
        [env_fn(f"configs/{env_config[i % len(env_config)]}.yaml", 0, i, args.capture_video, run_name) for
         i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = LSTMPPONetwork(envs).to(device)

    if ckpt_path is not None:
        agent.load_checkpoint(ckpt_path)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),

    )

    average_episode_reward = []
    for update in range(1, num_updates + 1):

        if global_step > total_steps:
            break

        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        not_printed_this_update = True

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state,
                                                                                        next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():

                    average_episode_reward.append(item['episode']['r'])
                    if not_printed_this_update:
                        print(
                            f"global_step={global_step}, episodic_return={np.mean(np.array(average_episode_reward)):.2f}")
                        average_episode_reward = []
                        not_printed_this_update = False
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if update % 10 == 0:
            print("SPS:", int(global_step / (time.time() - start_time)))
            print(f"Saving checkpoint, global step: {global_step}")
            agent.save_checkpoint(path=f"weights_{env_config}.ckpt")
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    print(f"Saving checkpoint, global step: {global_step}")
    agent.save_checkpoint(path=f"weights_{env_config}.ckpt")

    envs.close()
    writer.close()


def test(env_name, ckpt_path, pcg, total_num_episodes):
    env_config = "cluster-5.yaml"
    ckpt_path = "weights_['cluster-1', 'cluster-2', 'cluster-3', 'cluster-4'].ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pcg:
        env = make_pcg_env(f"configs/{env_config}", 0, 0, 0, 0)()
    else:
        env = make_env(env_name)(f"configs/{env_config}", 0, 0, 0, 0)
    env.single_action_space = env.action_space

    agent = LSTMPPONetwork(env).to(device)

    agent.load_checkpoint(ckpt_path)
    video_recorder = VideoRecorder()

    obs = env.reset()

    video_frame = np.moveaxis(obs, 0, -1)
    video_recorder.start(f"videos/lstm/{ckpt_path.split('.')[0]}_{env_config.split('.')[0]}.mp4", video_frame.shape)

    obs = torch.tensor(obs).to(device)
    total_reward = 0
    num_episodes = 0
    episode_rewards = []
    average_episode_reward = []

    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),

    )
    next_done = torch.zeros(1, ).to(device)
    while num_episodes < total_num_episodes:

        action, _, _, _, next_lstm_state = agent.get_action_and_value(obs.unsqueeze(0), next_lstm_state, next_done)
        obs, reward, done, info = env.step(action.cpu().numpy())
        if num_episodes < 10:
            video_frame = np.moveaxis(obs, 0, -1)
            video_recorder.add_frame(video_frame)

        obs = torch.from_numpy(obs).to(device)

        next_done = torch.Tensor(np.array([done]).astype(int)).to(device)
        # env.render()
        total_reward += reward
        if done:
            num_episodes += 1
            episode_rewards.append(total_reward)
            total_reward = 0
            env.reset()
            if num_episodes % 10 == 0:
                print("Episode: ", num_episodes)

            if "episode" in info.keys():
                average_episode_reward.append(info['episode']['r'])

    video_recorder.close()
    print("Average episode rewards: ", np.array(average_episode_reward).mean())
    print("Episode rewards: ", np.array(episode_rewards).mean())


