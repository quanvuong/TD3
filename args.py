import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    return args
