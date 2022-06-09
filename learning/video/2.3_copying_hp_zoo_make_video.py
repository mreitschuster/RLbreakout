#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

name_model='2.3_copying_hp_zoo'

import os

model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/best_model')

env_id                = 'Breakout-v4'
policy                = 'CnnPolicy'

#%%

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper # this includes EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage


def create_env(env_id, n_envs, seed, frame_stack):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         seed          = seed,
                         wrapper_class = AtariWrapper,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    
env = create_env(env_id=env_id, n_envs=1, seed=seed, frame_stack=4)


from stable_baselines3 import PPO

model = PPO(policy,env) 
model.load(model_folder)

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

    if done.any():
#    if done:   
        print('final reward:' + str(cum_reward))
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