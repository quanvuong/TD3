# Addressing Function Approximation Error in Actor-Critic Methods

A port of the official PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3) to Python 3 with a few additional changes:

1. trange from the tqdm package is used to display progress bar
2. torch.set_num_threads(1)
3. Use tensorboardX to visualize evaluation performance during training
4. One centralized function to set seed. Also set the seed of the GPU.
5. Evaluation is run for 100 instead of 10 episodes.

### Usage
The results can be reproduced exactly by running:
```
./experiments.sh
```
Experiments on single environments can be run by calling:
```
python main.py --env HalfCheetah-v2
```

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DDPG (DDPG.py) for easy comparison of hyper-parameters with TD3, this is not the implementation of "Our DDPG" as used in the paper (see OurDDPG.py). 

Algorithms which TD3 compares against (PPO, TRPO, ACKTR, DDPG) can be found at [OpenAI baselines repository](https://github.com/openai/baselines). 

### Results
Learning curves found in the paper are found under /learning_curves. Each learning curve are formatted as NumPy arrays of 201 evaluations (201,), where each evaluation corresponds to the average total reward from running the policy for 10 episodes with no exploration. The first evaluation is the randomly initialized policy network (unused in the paper). Evaluations are peformed every 5000 time steps, over a total of 1 million time steps. 

Numerical results can be found in the paper, or from the learning curves. Video of the learned agent can be found [here](https://youtu.be/x33Vw-6vzso). 
