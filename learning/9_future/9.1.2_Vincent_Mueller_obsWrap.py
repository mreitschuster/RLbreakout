#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
code reporuding https://towardsdatascience.com/training-rl-agents-in-stable-baselines3-is-easy-9d01be04c9db
"""

name_model='9.1_VincentMueller'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# env
env_id                = 'Breakout-v4'
n_envs                = 8

# eval
seed=123




#%%
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import A2C

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
n_envs=16
n_envs_eval=4

from BreakoutWrapper import wrapper_class_generator, create_env

flag_col     = 'mono_1dim'        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
flag_dim     = 'trim'        # 'blacken', 'whiten', 'keep', 'trim'
flag_predict = 'predict'   # 'nopredict' , 'predict' , 'predict_counters'
flag_FireResetEnv = True
frame_stack = 3
MaxAndSkipEnv_skip = 0
env_id                = 'BreakoutNoFrameskip-v4'
#env_id                = 'Breakout-v4'
instance_wrapper_class_train  =wrapper_class_generator(flag_col    = flag_col,
                                                       flag_dim    = flag_dim,
                                                       flag_predict = flag_predict,
                                                       flag_EpisodicLifeEnv = True,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)
instance_wrapper_class_eval   =wrapper_class_generator(flag_col    = flag_col,
                                                       flag_dim    = flag_dim,
                                                       flag_predict = flag_predict,
                                                       flag_EpisodicLifeEnv = True,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)

train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_train , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_eval, n_envs=n_envs_eval, seed=seed+1, frame_stack=frame_stack)


#%%
from stable_baselines3.common.callbacks import EvalCallback
eval_freq=50000
n_eval_episodes=5
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=model_folder,
                             n_eval_episodes=n_eval_episodes,
                             eval_freq=max(eval_freq // n_envs_eval, 1),
                             deterministic=False, 
                             render=False) 


#%%

# Frame-stacking with 4 frames
model = A2C("CnnPolicy", train_env, verbose=1,
            seed            = seed,
            tensorboard_log = tensorboard_folder)
model.learn(total_timesteps=int(5e6), callback=eval_callback)

train_env.close()
eval_env.close()

#model = A2C.load("A2C_breakout") #uncomment to load saved model

