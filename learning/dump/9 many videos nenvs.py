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
    
train_env = create_env(env_id=env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack)


#%% Loading existing baseline model

from stable_baselines3 import PPO
model = PPO(policy, train_env)
baselinemodel=os.path.expanduser('~/models/breakout-v4/model/2.3_copying_hp_zoo/best_model.zip')
assert os.path.exists(baselinemodel) # if it doesnt exist go back to 2.3_copying_hp_zoo.py
model.load(baselinemodel)


#%% Let's see how it plays
import time

state = train_env.reset()

for step in range(int(1e3)):
    # do something useful
    action, _ = model.predict(state)
    state, reward, done, info = train_env.step(action)
    
    image=train_env.render(mode='human')
    time.sleep(0.1)

    if done.all():
        print('final reward:' + str(reward))
        break
        train_env.reset()
        

# Close the env
# only this seems to be able to close the window in which the game was rendered
train_env.close()