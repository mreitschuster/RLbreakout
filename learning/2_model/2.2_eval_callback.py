#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

#%%

from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage


def create_env(env_id,n_envs=1):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         wrapper_class = EpisodicLifeEnv)   

    new_env = VecTransposeImage(new_env)     
    return new_env

env=create_env('Breakout-v4',8)
eval_env=create_env('Breakout-v4',2)

#%%
import os
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=os.path.expanduser('~/models/breakout-v4/2.2_eval_callback/')) 


#%% Training
from stable_baselines3 import PPO


model = PPO(policy          = 'CnnPolicy', 
            env             = env,     
            verbose         = 1, 
            seed            = seed,
            tensorboard_log = os.path.expanduser('~/models/breakout-v4/tb_log/'))

model.learn(total_timesteps = 2e5,
            callback        = eval_callback, 
            tb_log_name     = '2.2_eval_callback')

env.close()