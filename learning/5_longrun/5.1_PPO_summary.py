#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='5.1_PPO_summary_1'
study_name=name_model


import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# eval
seed=123

#%%
import os
from create_objective_shedule import create_objective

#%%
# Creating the first Study 

instance_objective = create_objective(N_EVAL_EPISODES    = 30, 
                                       EVAL_FREQ          = 40, 
                                       TRAINING_STEPS     = 1e7, 
                                       n_envs_eval        = 8,
                                       study_name         = study_name,
                                       model_folder       = model_folder,
                                       tensorboard_folder = tensorboard_folder,
                                       verbose            = 1,
                                       risk_adjustment_stds=1,
                                       N_Rank=int(30/4),    # use lower quartile
                                       seed=seed
                                       )

if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
import yaml
dbconnector = yaml.safe_load(open( os.path.expanduser('~/optunaDB.yaml')))['dbconnector']

import optuna
pruner=optuna.pruners.NopPruner()
study1 = optuna.create_study(direction='maximize', storage=dbconnector, study_name=study_name, load_if_exists=True, pruner=pruner)

import math
if False:
#if True:
    # base
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', # number 7
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 0.001,
                          'clip_range_init':    .2,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'delta': math.log(0.001/0.0001)/1e6 # 0.1^10 =1e-10
                          })
    

    study1.enqueue_trial({'train_env_id': 'Breakout-v4', # number 8
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 0.001,
                          'clip_range_init':    .2,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'delta': math.log(0.001/0.0005)/1e6 # 0.5^10 =0.1
                          })
    
    study1.enqueue_trial({'train_env_id': 'Breakout-v4',  # number 9
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 0.001,
                          'clip_range_init':    .2,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'delta': math.log(0.001/0.00055)/1e6 # 0.55^10 =0.25
                          })
    study1.enqueue_trial({'train_env_id': 'Breakout-v4',  # number 10
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 5e-4,
                          'clip_range_init':    .2,
                          'lr_shedule': 'linear',
                          'clip_shedule': 'linear',
                          'delta': math.log(0.001/0.00055)/1e6 # 0.55^10 =0.25
                          })
    
if True: 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4',  # number 11
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 0.001,
                          'clip_range_init':    .2,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'delta': math.log(0.001/0.00063)/1e6 # 0.63^10 =1 -> end LR is aroud 1e-5 which is where clip rate drops to zero in other runs
                          })
    
    

    pass
    
#study1.optimize(instance_objective)