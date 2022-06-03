#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

name_model='1.3_train'

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


baselinemodel=os.path.expanduser('~/models/breakout-v4/model/1.3_train_make_video/best_model')
if os.path.exists(baselinemodel+'.zip'):
    model.load(baselinemodel)
else:
    model.learn(total_timesteps = 1e6,
            tb_log_name     = '1.3_train_make_video')
    model.save(baselinemodel)

#%% Let's see how it plays

from gym.wrappers.monitoring.video_recorder import VideoRecorder
video_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.mp4')
gif_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.gif')
video_recorder = VideoRecorder(env, video_file, enabled=True)

state=env.reset()
cum_reward=0
for step in range(int(1e3)):
    # do something useful
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    cum_reward=cum_reward+reward
    image=env.render()
    video_recorder.capture_frame()
    #time.sleep(0.1)

    if done:
        print('final reward:' + str(reward))
        cum_reward=0
    #    break
        env.reset()
        
        
video_recorder.close()
# Close the env
# only this seems to be able to close the window in which the game was rendered
env.close()


#%%
cmd1='ffmpeg -i '
cmd2=' -r 10 -f image2pipe -vcodec ppm - | convert -delay 10 -loop 0 -layers Optimize - '
cmd=cmd1+video_file+cmd2+gif_file

os.system(cmd)