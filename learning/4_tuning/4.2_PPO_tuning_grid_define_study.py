#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.2_PPO_grid_3'
study_name=name_model

import os
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')

#%%
# Creating the first Study 


if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
import yaml
dbconnector = yaml.safe_load(open( os.path.expanduser('~/optunaDB.yaml')))['dbconnector']

import optuna
study1 = optuna.create_study(direction='maximize', storage=dbconnector, study_name=study_name, load_if_exists=True)


# to this only once
if False:
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 128, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 256, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 512, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 2048, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 4096, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3.5*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 2.5*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})

    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 2, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 3, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 5, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 6, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 7, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 8, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 9, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': 3*1024, 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 10, 'n_envs': 8})
if True: 
    
    pass
if False:
    pass 
