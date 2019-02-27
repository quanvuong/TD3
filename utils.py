import time
import random

import itertools
import copy
import numpy as np
# Expects tuples of (state, next_state, action, reward, done)
import torch


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

def chunk(iterable, **kwargs):
    chunk_size = kwargs.pop('chunk_size', 50)
    it = iter(iterable)
    step = 0
    while True:
       chunk = list(itertools.islice(it, chunk_size))
       if len(chunk) == 0:
           return
       yield chunk, slice(step, step + len(chunk))
       step += len(chunk)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

#@timeit
def evaluate_policy(policy, env, eval_episodes=100, env_chunk_size=50):
    #TODO: this function can be substantially speed up
    #This function currently consumes ~30-50% of compute time

    envs = [copy.deepcopy(env) for _ in range(eval_episodes)]
    for i, env in enumerate(envs):
        env.seed(i)
    done = np.array([False] * eval_episodes)
    avg_reward = np.array([0.] * eval_episodes)

    for env_chunk, idx in chunk(envs, chunk_size=env_chunk_size):
        obs = np.stack([env.reset() for env in env_chunk])
        while not all(done[idx]):
            actions = policy.select_actions(obs)
            for i, (env, action, j) in enumerate(zip(env_chunk, actions, range(eval_episodes)[idx])):
                obs[i], r, d, _ = env.step(action)
                avg_reward[j] += 0 if done[j] else r
                done[j] = d

    avg_reward = np.mean(avg_reward)

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward
