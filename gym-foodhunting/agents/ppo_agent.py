# -*- coding: utf-8 -*-

import argparse
import gym
import gym_foodhunting

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2


def train(filename, total_timesteps, n_cpu):
    env_name = 'FoodHuntingContinuous-v0'
    # env_name = 'FoodHunting-v0'
    policy = CnnPolicy
    print(env_name, policy)
    #env = DummyVecEnv([lambda: gym.make(env_name)])
    # This needs to run to use SubprocVecEnv on Mac OS X 10.13.6.
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    # https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331
    env = SubprocVecEnv([lambda: gym.make(env_name) for i in range(n_cpu)])
    model = PPO2(policy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps, log_interval=2)
    model.save(filename)
    env.close()

def play(filename, total_timesteps):
    env_name = 'FoodHuntingContinuousGUI-v0'
    # env_name = 'FoodHuntingGUI-v0'
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
    parser.add_argument('--n_cpu', type=int, default=4, help='number of CPU.')
    parser.add_argument('--play', action='store_true', help='play or not.')
    args = parser.parse_args()

    if args.play:
        play(args.filename, args.total_timesteps)
    else:
        train(args.filename, args.total_timesteps, args.n_cpu)
