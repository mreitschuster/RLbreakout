#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

import gym
import time
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv


env = gym.make('Breakout-v4')
env = EpisodicLifeEnv(env)

state=env.reset()

for step in range(int(1e3)):
    # just do anythin
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print('reward: '+str(reward))
    print('done: '  +str(done))
    
    image=env.render()

    time.sleep(0.1)

    # If the epsiode is up, then start another one
    if done:
        # it never ends...
        print('final reward:' + str(reward))
        break
        env.reset()
        

# Close the env
# only this seems to be able to close the window in which the game was rendered
env.close()