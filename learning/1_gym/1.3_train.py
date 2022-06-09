#!/usr/bin/env python3
# -*- coding: utf-8 -*-

seed=123

import gym
import time
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv 

env = gym.make('Breakout-v4')
env = EpisodicLifeEnv(env)


#%% Training

from stable_baselines3 import PPO
import os

model = PPO(policy          = 'CnnPolicy', 
            env             = env,     
            verbose         = 1, 
            seed            = seed,
            tensorboard_log = os.path.expanduser('~/models/breakout-v4/tb_log/'))

model.learn(total_timesteps = 1e6,
            tb_log_name     = '1.3_train')


#%% Let's see how it plays

obs=env.reset()

for step in range(int(1e3)):

    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    time.sleep(0.1)

    if done:
        print('final reward:' + str(reward))
        break
        env.reset()
        
env.close()