#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:29:24 2022

@author: mreitschuster
"""

name_model='newModel'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/logs/')
model_folder=os.path.expanduser('~/models/breakout-v4/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/')

# env
env_id                = 'Breakout-v4'
n_envs                = 8

# model
algo                  = 'ppo'
policy                = 'CnnPolicy'
#policy                = 'MlpPolicy' # needed as not a picture anymore?!
n_steps               = 128
n_epochs              = 4
batch_size            = 256
n_timesteps           = 1e7
learning_rate_initial = 2.5e-4
clip_range_initial    = 0.1
vf_coef               = 0.5
ent_coef              = 0.01

# eval
seed=123
n_eval_episodes=5
n_eval_envs=1
eval_freq=25000

# hyper
frame_stack = 4
flag_plot=False
flag_grey=True
flag_trim=False

prediction_colour=[255,255,255]
prediction_height=3
prediction_width=16

flag_FireResetEnv=False
flag_EpisodicLifeEnv=False
flag_ClipRewardEnv=False
MaxAndSkipEnv_skip=0


# debug
#n_envs = 1
#flag_plot=True




#%%
import gym
from breakout_wrapper import Breakout2dObservationWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper,ClipRewardEnv,EpisodicLifeEnv,MaxAndSkipEnv, FireResetEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage

def wrapper_class(env):
    env1 = Breakout2dObservationWrapper(env, 
                                 flag_plot = flag_plot, 
                                 flag_grey = flag_grey, 
                                 flag_trim = flag_trim,
                                 prediction_colour = prediction_colour,
                                 prediction_height = prediction_height,
                                 prediction_width  = prediction_width)
    if flag_FireResetEnv:
        env1 = FireResetEnv(env1)
    if flag_EpisodicLifeEnv:
        env1 = EpisodicLifeEnv(env1)
    if flag_ClipRewardEnv:
        env1 = ClipRewardEnv(env1)
    if MaxAndSkipEnv_skip>0:
        env1=MaxAndSkipEnv(env1, skip=MaxAndSkipEnv_skip)
    
    return(env1)


def create_env(env_id,n_envs=1,seed=123,frame_stack=4):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         seed          = seed,
                         wrapper_class = wrapper_class,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    
train_env = create_env(env_id=env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack)

#%%
from stable_baselines3.common.callbacks import BaseCallback
class check_precision_callback(BaseCallback):

    def __init__(self, verbose=0):
        self.landing_list=[]
        self.last_recorded_step=0
        super(check_precision_callback, self).__init__(verbose)

    def _on_step(self):
        return True

precision_callback = check_precision_callback()
    
#%%
from stable_baselines3.common.callbacks import EvalCallback
# missing usage of SaveVecNormalizeCallback

eval_callback = EvalCallback(create_env(env_id=env_id, n_envs=n_eval_envs, seed=seed, frame_stack=frame_stack),
                             best_model_save_path=log_folder,
                             n_eval_episodes=n_eval_episodes,
                             log_path=log_folder, 
                             eval_freq=max(eval_freq // n_envs, 1),
                             deterministic=False, 
                             render=False) # see exp_manager.py line 448

#%%
# create learning rate and clip rate functions
# see _preprocess_hyperparams() line 168 in exp_manager.py
# which uses _preprocess_schedules() line 286 in exp_manager.py
# which uses linear_schedule() line 256 in utils.py
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

learning_rate_shedule = linear_schedule(learning_rate_initial)
clip_range_shedule    = linear_schedule(clip_range_initial)

#%%

from stable_baselines3 import PPO
import torch

model = PPO(policy, 
            train_env, 
            n_steps        = n_steps,
            n_epochs       = n_epochs,
            batch_size     = batch_size,
            learning_rate  = learning_rate_shedule,
            clip_range     = clip_range_shedule,
            vf_coef        = vf_coef,
            ent_coef       = ent_coef,            
            verbose        = 1, 
            seed            = seed,
            tensorboard_log = tensorboard_folder) # exp_manager.py line 185


#%%
model.learn(total_timesteps = n_timesteps,
            #callback        = [eval_callback, precision_callback], 
            callback        = eval_callback, 
            tb_log_name     = name_model)

#%%
model.save(model_folder+name_model)