# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import gym
import gym_foodhunting

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

BEST_SAVE_FILE_SUFFIX = '_best'

def make_env(env_name, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def learn(env_name, load_file, save_file, total_timesteps, n_cpu, reward_threshold):
    best_mean_reward = -np.inf
    best_mean_step = np.inf
    counter = 0
    best_save_file = save_file + BEST_SAVE_FILE_SUFFIX
    start_time = time.time()
    def callback(_locals, _globals):
        nonlocal best_mean_reward, best_mean_step, counter
        counter += 1
        if counter % 10 == 0:
            t = int((time.time() - start_time) / 60.0)
            print('minutes:', t)
        ep_info_buf = _locals['ep_info_buf']
        if len(ep_info_buf) < ep_info_buf.maxlen:
            return True
        mean_reward = np.mean([ ep_info['r'] for ep_info in ep_info_buf ])
        mean_step = np.mean([ ep_info['l'] for ep_info in ep_info_buf ])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print('best_mean_reward:', best_mean_reward)
            print('saving new best model:', best_save_file)
            _locals['self'].save(best_save_file)
        if mean_step < best_mean_step:
            best_mean_step = mean_step
            print('best_mean_step:', best_mean_step)
        return mean_reward < reward_threshold # False should finish learning
    # policy = CnnPolicy
    policy = CnnLstmPolicy
    print(env_name, policy)
    # Run this to enable SubprocVecEnv on Mac OS X.
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    # see https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331
    env = SubprocVecEnv([make_env(env_name, i) for i in range(n_cpu)])
    if load_file is not None:
        model = PPO2.load(load_file, env, verbose=1)
    else:
        model = PPO2(policy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=5, callback=callback)
    print('saving model:', save_file)
    model.save(save_file)
    env.close()

def reshapeObservation(obss, n_cpu):
    # before: (    1, 64, 64, 4)
    # after:  (n_cpu, 64, 64, 4)
    for i in range(n_cpu-1):
        obss = np.insert(obss, [i+1], obss[0], axis=0)
    return obss

def play(env_name, load_file, total_timesteps, n_cpu):
    isGUI = env_name.find('GUI') != -1 # if it's GUI mode, number of env is changed to 1. but trained LSTM model cannot change number of env. so it needs to reshape observation.
    model = PPO2.load(load_file, verbose=1)
    env = SubprocVecEnv([make_env(env_name, i) for i in range(1 if isGUI else n_cpu)])
    obss = env.reset()
    obss = reshapeObservation(obss, n_cpu) if isGUI and n_cpu != 1 else obss
    rewards_buf = []
    steps_buf = []
    for i in range(total_timesteps):
        actions, _states = model.predict(obss)
        actions = actions[0:1] if isGUI else actions
        obss, rewards, dones, infos = env.step(actions)
        obss = reshapeObservation(obss, n_cpu) if isGUI and n_cpu != 1 else obss
        # env.render() # dummy
        if dones.any():
            rewards_buf.extend([ info['episode']['r'] for info in infos if 'episode' in info ])
            steps_buf.extend([ info['episode']['l'] for info in infos if 'episode' in info ])
            print('mean:', np.mean(rewards_buf), np.mean(steps_buf))
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='play or learn.')
    parser.add_argument('--env_name', type=str, default='FoodHunting-v0', help='environment name.')
    parser.add_argument('--load_file', type=str, default=None, help='filename to load model.')
    parser.add_argument('--save_file', type=str, default='foodhunting', help='filename to save model.')
    parser.add_argument('--total_timesteps', type=int, default=500000, help='total timesteps.')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of CPU cores.')
    parser.add_argument('--reward_threshold', type=float, default=3.0, help='reward threshold to finish learning.')
    args = parser.parse_args()

    if args.play:
        play(args.env_name, args.load_file, args.total_timesteps, args.n_cpu)
    else:
        learn(args.env_name, args.load_file, args.save_file, args.total_timesteps, args.n_cpu, args.reward_threshold)
