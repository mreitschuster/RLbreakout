#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.2_PPO_grid_4'
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
    
    #Base
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int((2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    #iterate batch size
    
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(2**7), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(2**8), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    #base-scenario
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(2*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(3*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(4*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(5*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(7*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    
    # iterate n_steps
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int((2**9)), 'n_steps': int(1*64), 'frame_stack': 4, 'frameskip_env': 2, 'n_envs': 8})
    #base-scenario
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int((2**9)), 'n_steps': int(3*64), 'frame_stack': 4, 'frameskip_env': 2, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int((2**9)), 'n_steps': int(4*64), 'frame_stack': 4, 'frameskip_env': 2, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int((2**9)), 'n_steps': int(8*64), 'frame_stack': 4, 'frameskip_env': 2, 'n_envs': 8})
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int((2**9)), 'n_steps': int(16*64), 'frame_stack': 4, 'frameskip_env': 2, 'n_envs': 8})
    
    # batch/steps diag
    base_batch=(2**9)
    base_n_steps=(2*64)
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(2*base_batch), 'n_steps': int(2*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(3*base_batch), 'n_steps': int(3*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(4*base_batch), 'n_steps': int(4*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(5*base_batch), 'n_steps': int(5*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(6*base_batch), 'n_steps': int(6*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(7*base_batch), 'n_steps': int(7*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 4, 'batch_size': int(8*base_batch), 'n_steps': int(8*base_n_steps), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    
    # iterate epochs
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 1, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 2, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 3, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    #base
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 5, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 6, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 7, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 8, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 10, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 12, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 16, 'batch_size': int(6*(2**9)), 'n_steps': 128, 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    
    # expanding the epochs/batch field
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 6, 'batch_size': int((2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 8, 'batch_size': int((2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 6, 'batch_size': int(2*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 8, 'batch_size': int(2*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})     
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 6, 'batch_size': int(4*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 8, 'batch_size': int(4*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})   
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 6, 'batch_size': int(6*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 8, 'batch_size': int(6*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})   
    
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 6, 'batch_size': int(8*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 'n_epochs': 8, 'batch_size': int(8*(2**9)), 'n_steps': int(2*64), 'frame_stack': 4, 'frameskip_env': 4, 'n_envs': 8})  
if True: 

    
    pass
if False:
    pass 
