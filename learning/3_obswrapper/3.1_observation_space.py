#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

import os
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/')
name_model='3.1_observation_space'

# env
env_id                = 'Breakout-v4'
n_envs                = 8
frame_stack           = 4

# model
algo                  = 'ppo'
policy                = 'CnnPolicy'
n_steps               = 128
n_epochs              = 4
batch_size            = 256
n_timesteps           = 1e7
learning_rate_initial = 2.5e-4
clip_range_initial    = 0.1
vf_coef               = 0.5
ent_coef              = 0.01

#%% we need to recreate the environment, as it is not saved with the model

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
    
train_env = create_env(env_id=env_id, n_envs=1, seed=seed, frame_stack=frame_stack)


#%% Loading existing baseline model

from stable_baselines3 import PPO
model = PPO(policy, train_env)
baselinemodel=os.path.expanduser('~/models/breakout-v4/model/2.3_copying_hp_zoo/best_model.zip')
assert os.path.exists(baselinemodel) # if it doesnt exist go back to 2.3_copying_hp_zoo.py
model.load(baselinemodel)


#%% Let's see how it plays
import time
import numpy as np
state = train_env.reset()
image=train_env.render(mode='rgb_array')

print(state.shape) # (1,4,84,84)   4 is framestack
print(image.shape) # (210,160,3)   3 is colour channels

#def prep_state(state):
#    image_state=np.stack([state[0,0,:,:],state[0,0,:,:],state[0,0,:,:]],axis=2) # we stack the 1-colour channel 3 times to have a grey image in rgb
#    return image_state

for step in range(int(30)):

    action, _ = model.predict(state)
    state, reward, done, info = train_env.step(action) # state is the picture after wrappers
    image=train_env.render(mode='rgb_array')    # we want tp have access to the image of the underlying environment. 
    
from PIL import Image
im1 = Image.fromarray(state[0,0,:,:])
im1.save(os.path.expanduser('~/models/breakout-v4/image/3.1_observation_space_afterWrapper.jpeg'))
im2 = Image.fromarray(image)
im2.save(os.path.expanduser('~/models/breakout-v4/image/3.1_observation_space_beforeWrapper.jpeg'))
        

# Close the env
# only this seems to be able to close the window in which the game was rendered
train_env.close()