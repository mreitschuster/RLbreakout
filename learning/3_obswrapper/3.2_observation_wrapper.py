#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

import os
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/')
name_model='3.2_observation_wrapper'
image_folder=os.path.expanduser('~/models/breakout-v4/image/')


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


#%% new observation wrapper
import gym
import numpy as np
class BreakoutObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, 
                 flag_grey=True, 
                 flag_trim=True,
                 flag_mono=True
                 ):
        
        self.threshold_color=50
        
        self.screen_boundary_left = 8
        self.screen_boundary_right = 151
        
        self.ball_freepane_row_upper = 93
        self.ball_freepane_row_lower = 188
        
        self.padpane_row_upper = 189
        self.padpane_row_lower = 192
        
        self.start_row = 32
        self.end_row   = self.padpane_row_lower
                
        super().__init__(env)
        
        if flag_trim:
            rows=self.end_row  - self.start_row
            cols=self.screen_boundary_right-self.screen_boundary_left
        else:
            rows=210
            cols=160
            
        nr_colours=3
        if flag_grey:
            nr_colours=1
                
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(rows, cols,
                                                       nr_colours), 
                                                dtype=np.uint8)
            
        self.flag_grey=flag_grey
        self.flag_trim=flag_trim
        self.flag_mono=flag_mono
    
    def observation(self, obs):
        
        # trim the image to relevant zones
        if self.flag_trim:
            image_cut = obs[self.start_row:self.end_row, 
                            self.screen_boundary_left:self.screen_boundary_right,
                            :]
        else :
            image_cut = obs
        
        if self.flag_mono:
            # mono beats grey, as mono is also a kind of grey
            image_cut_Ncol = (np.amax(image_cut,2, keepdims=True)>self.threshold_color)*1.
        elif self.flag_grey:
            image_cut_Ncol = np.amax(image_cut,2, keepdims=True)
        else:
            image_cut_Ncol = image_cut
            
        return image_cut_Ncol


#%% we need to recreate the environment, as it is not saved with the model

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage

def create_env(env_id, n_envs, seed, frame_stack):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         seed          = seed,
                         wrapper_class = BreakoutObservationWrapper,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
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

for step in range(int(23)): # we just want some in game pic

    action, _ = model.predict(state)
    state, reward, done, info = train_env.step(action) # state is the picture after wrappers
    image=train_env.render(mode='rgb_array')    # we want tp have access to the image of the underlying environment. 
    
train_env.close()

#%%
from PIL import Image

if state.shape[1]==12: # color + framestack on same dimension
    arr_state=np.transpose(state[0,9:12,:,:],(1,2,0))
elif state.shape[1]==4: # greyscale. 
    arr_state=state[0,3,:,:] # the last of the 4 elements in the second dimension corresponds to current. the others are past.
    if np.amax(arr_state)==1: # monoscale
        arr_state=arr_state*255
else:
    raise NameError("i did not understand the dimensions of the array.")


im1 = Image.fromarray(arr_state) 
im1.save(image_folder+name_model+'_afterWrapper.jpeg')

im2 = Image.fromarray(image)
im2.save(image_folder+name_model+'_beforeWrapper.jpeg')
        
