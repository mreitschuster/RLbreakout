#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

#%%

from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv

env=make_vec_env(env_id        = 'Breakout-v4', 
                     n_envs        = 8, 
                     wrapper_class = EpisodicLifeEnv
                     )    


#%% Training

from stable_baselines3 import PPO
import os

model = PPO(policy          = 'CnnPolicy', 
            env             = env,     
            verbose         = 1, 
            seed            = seed,
            tensorboard_log = os.path.expanduser('~/models/breakout-v4/tb_log/'))

model.learn(total_timesteps = 2e5,
            tb_log_name     = '2.1_envs')


env.close()