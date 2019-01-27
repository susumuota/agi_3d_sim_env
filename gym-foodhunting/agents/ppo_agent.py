# -*- coding: utf-8 -*-

import argparse
import numpy as np
import gym
import gym_foodhunting

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

def get_env_name(discrete, render):
    name = 'FoodHunting'
    name = name + 'Discrete' if discrete else name
    name = name + 'GUI' if render else name
    name = name + '-v0'
    return name

def train(filename, total_timesteps, discrete, n_cpu, reward_threshold):
    best_mean_reward = -np.inf
    best_mean_step = np.inf
    def callback(_locals, _globals):
        nonlocal best_mean_reward, best_mean_step
        ep_info_buf = _locals['ep_info_buf']
        mean_reward = np.mean([ ep_info['r'] for ep_info in ep_info_buf ])
        mean_step = np.mean([ ep_info['l'] for ep_info in ep_info_buf ])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print('best_mean_reward:', best_mean_reward)
            print('saving new best model:', filename)
            _locals['self'].save(filename)
        if mean_step < best_mean_step:
            best_mean_step = mean_step
            print('best_mean_step:', best_mean_step)
        return False if mean_reward >= reward_threshold else True
    env_name = get_env_name(discrete, False)
    policy = CnnPolicy
    print(env_name, policy)
    # Run this to enable SubprocVecEnv on Mac OS X.
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    # see https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331
    env = SubprocVecEnv([lambda: gym.make(env_name) for i in range(n_cpu)])
    model = PPO2(policy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=2, callback=callback)
    model.save(filename)
    env.close()

def play(filename, total_timesteps, discrete):
    env_name = get_env_name(discrete, True)
    policy = CnnPolicy
    print(env_name, policy)
    env = DummyVecEnv([lambda: gym.make(env_name)])
    model = PPO2.load(filename, env, verbose=1)
    obs = env.reset()
    for i in range(total_timesteps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='foodhunting', help='filename.')
    parser.add_argument('--total_timesteps', type=int, default=50000, help='total timesteps.')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of CPU cores.')
    parser.add_argument('--reward_threshold', type=float, default=3.0, help='reward threshold to finish training.')
    parser.add_argument('--discrete', action='store_true', help='discrete or continuous action.')
    parser.add_argument('--play', action='store_true', help='play or not.')
    args = parser.parse_args()

    if args.play:
        play(args.filename, args.total_timesteps, args.discrete)
    else:
        train(args.filename, args.total_timesteps, args.discrete, args.n_cpu, args.reward_threshold)
