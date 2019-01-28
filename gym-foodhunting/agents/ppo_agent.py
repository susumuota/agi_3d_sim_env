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

def learn(filename, total_timesteps, discrete, render, n_cpu, reward_threshold, step_threshold):
    best_mean_reward = -np.inf
    best_mean_step = np.inf
    def callback(_locals, _globals):
        nonlocal best_mean_reward, best_mean_step
        ep_info_buf = _locals['ep_info_buf']
        mean_reward = np.mean([ ep_info['r'] for ep_info in ep_info_buf ])
        mean_step = np.mean([ ep_info['l'] for ep_info in ep_info_buf ])
        print('mean:', mean_reward, mean_step)
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print('best_mean_reward:', best_mean_reward)
            print('saving new best model:', filename)
            _locals['self'].save(filename)
        if mean_step < best_mean_step:
            best_mean_step = mean_step
            print('best_mean_step:', best_mean_step)
            if mean_reward >= reward_threshold:
                print('saving new best model:', filename)
                _locals['self'].save(filename)
        return mean_reward < reward_threshold or mean_step > step_threshold # False should finish learning
    env_name = get_env_name(discrete, render)
    policy = CnnPolicy
    print(env_name, policy)
    # Run this to enable SubprocVecEnv on Mac OS X.
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    # see https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331
    env = SubprocVecEnv([lambda: gym.make(env_name) for i in range(n_cpu)])
    model = PPO2(policy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=5, callback=callback)
    model.save(filename)
    env.close()

def play(filename, total_timesteps, discrete, render):
    env_name = get_env_name(discrete, render)
    policy = CnnPolicy
    print(env_name, policy)
    env = DummyVecEnv([lambda: gym.make(env_name)])
    model = PPO2.load(filename, env, verbose=1)
    obss = env.reset()
    rewards_buf = []
    steps_buf = []
    for i in range(total_timesteps):
        action, _states = model.predict(obss)
        obss, rewards, dones, infos = env.step(action)
        env.render() # dummy
        if dones.all():
            rewards_buf.append([ info['episode']['r'] for info in infos ])
            steps_buf.append([ info['episode']['l'] for info in infos ])
            print('mean:', np.mean(rewards_buf), np.mean(steps_buf))
            # obss = env.reset() # does not need this line but why?
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='foodhunting', help='filename.')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='total timesteps.')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of CPU cores.')
    parser.add_argument('--reward_threshold', type=float, default=3.0, help='reward threshold to finish learning.')
    parser.add_argument('--step_threshold', type=float, default=50.0, help='step threshold to finish learning.')
    parser.add_argument('--discrete', action='store_true', help='discrete or continuous action.')
    parser.add_argument('--render', action='store_true', help='render or not.')
    parser.add_argument('--play', action='store_true', help='play or learn.')
    args = parser.parse_args()

    if args.play:
        play(args.filename, args.total_timesteps, args.discrete, args.render)
    else:
        learn(args.filename, args.total_timesteps, args.discrete, args.render, args.n_cpu, args.reward_threshold, args.step_threshold)
