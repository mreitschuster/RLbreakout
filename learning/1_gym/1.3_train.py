#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
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

model.learn(total_timesteps = 2e5,
            tb_log_name     = '1.3_train')


#%% Let's see how it plays

state=env.reset()

for step in range(int(1e3)):
    # do something useful
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    
    image=env.render()
    time.sleep(0.1)

    if done:
        print('final reward:' + str(reward))
        break
        env.reset()
        

# Close the env
# only this seems to be able to close the window in which the game was rendered
env.close()