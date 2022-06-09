#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import time
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv


env = gym.make('Breakout-v4', render_mode='human')
env = EpisodicLifeEnv(env)

state=env.reset()

for step in range(int(1e3)):
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print('reward: '+str(reward))
    print('done: '  +str(done))

    time.sleep(0.1)

    
    if done:
        # it never ends...
        print('final reward:' + str(reward))
        break
        env.reset()
        


env.close()