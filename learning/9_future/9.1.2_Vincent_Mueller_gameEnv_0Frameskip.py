#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
code reproducing https://towardsdatascience.com/training-rl-agents-in-stable-baselines3-is-easy-9d01be04c9db
"""

n_envs=16
frameskip=4
n_envs_eval=4

folder_name='9.1_Vincent_Mueller'
name_model='9.1.2_Vincent_Mueller_gameEnv_' + str(frameskip) + 'Frameskip_'+ str(n_envs) + 'envs'

import os
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+folder_name+'/')            # tb has seperate variable for name
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+folder_name+'/'+name_model+'/')    # 


# eval
seed=123


#%%
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, FireResetEnv, WarpFrame, EpisodicLifeEnv, MaxAndSkipEnv, NoopResetEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import A2C
import gym

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)


env_id='BreakoutNoFrameskip-v4'
#env_id='Breakout-v4'


def wrapper_class_generator(
                  flag_EpisodicLifeEnv,
                  MaxAndSkipEnv_skip):
    
    def wrap_env(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=30)
        if MaxAndSkipEnv_skip>0:
            env = MaxAndSkipEnv(env, skip=MaxAndSkipEnv_skip)
        if flag_EpisodicLifeEnv:
            env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env, width=84, height=84)
        env = ClipRewardEnv(env)
        
        return(env)

    return wrap_env # we return the function, not the result of the function



env = make_vec_env(env_id,  n_envs=n_envs, wrapper_class=wrapper_class_generator(flag_EpisodicLifeEnv=True,MaxAndSkipEnv_skip=frameskip))
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env) 

eval_env = make_vec_env(env_id, seed=seed, n_envs=n_envs, wrapper_class=wrapper_class_generator(flag_EpisodicLifeEnv=False,MaxAndSkipEnv_skip=frameskip))
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env) 
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
model = A2C("CnnPolicy", env, verbose=1,
            seed            = seed,
            tensorboard_log = tensorboard_folder)
model.learn(total_timesteps=int(1e8), callback=eval_callback, tb_log_name=name_model)

env.close()
eval_env.close()

#model = A2C.load("A2C_breakout") #uncomment to load saved model
model.save("A2C_breakout")
