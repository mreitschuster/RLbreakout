#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import time

env = gym.make('Breakout-v4', render_mode='human')

obs=env.reset()

for step in range(int(1e3)):
        
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    print('reward: '+str(reward))
    print('done: '  +str(done))

    time.sleep(0.1)

    if done:
        
        print('final reward:' + str(reward))
        break
        env.reset()
        

env.close()