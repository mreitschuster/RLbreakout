#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='2.3_copying_hp_zoo_1e6'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model)
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model)
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/')

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
n_timesteps           = 1e6
learning_rate_initial = 2.5e-4
clip_range_initial    = 0.1
vf_coef               = 0.5
ent_coef              = 0.01

# eval
seed=123
n_eval_episodes=5
n_eval_envs=1
eval_freq=25000


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
    
train_env = create_env(env_id=env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack)


#%%
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(create_env(env_id=env_id, n_envs=n_eval_envs, seed=seed, frame_stack=frame_stack),
                             best_model_save_path=model_folder,
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
from typing import  Callable, Union

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
            callback        = eval_callback, 
            tb_log_name     = name_model)

#%%
model.save(model_folder+name_model)