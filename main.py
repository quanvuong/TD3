import os

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

import DDPG
import OurDDPG
import TD3
import utils
from args import get_args
from utils import set_global_seeds, evaluate_policy


@profile
def main():

    torch.set_num_threads(1)

    args = get_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    writer = SummaryWriter(comment=file_name)

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    set_global_seeds(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    # evaluations = [evaluate_policy(policy, env)]
    # writer.add_scalar('episode_count/eval_performance', evaluations[0], 0)

    timesteps_since_eval = 0
    episode_num = 0
    done = True

    timestep_range = trange(int(args.max_timesteps))

    for total_timesteps in timestep_range:

        if done:
            episode_num += 1

            if total_timesteps != 0:
                timestep_range.set_postfix({
                    'Episode Num': episode_num,
                    'Episode Timesteps': episode_timesteps,
                    'Episode Reward': episode_reward
                })
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                 args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            # if timesteps_since_eval >= args.eval_freq:
            #     timesteps_since_eval %= args.eval_freq
            #
            #     eval_perf = evaluate_policy(policy, env)
            #     evaluations.append(eval_perf)
            #     writer.add_scalar('episode_count/eval_performance', eval_perf, episode_num)
            #
            #     if args.save_models: policy.save(file_name, directory="./pytorch_models")
            #     np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy, env))
    if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)


if __name__ == '__main__':
    main()
