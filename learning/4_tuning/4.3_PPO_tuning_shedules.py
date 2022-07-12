#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.3_PPO_tuning_shedules_10'
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
                                       EVAL_FREQ          = 4, 
                                       TRAINING_STEPS     = 1e6, 
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

if False:
#if True:
    # base
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 1e-3,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-2
                          })
    
    
    
    # varying lr init
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 1e-2,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-2
                          })
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 1e-1,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-2
                          })
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 1,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-2
                          })
    
    # varying clip init
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 10e-4,
                          'clip_range_init':    .2,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-2
                          })
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 10e-4,
                          'clip_range_init':    .3,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-2
                          })

    
    # varying target factor 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 10e-4,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-1
                          })
    
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 10e-4,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-3
                          })
    
    study1.enqueue_trial({'train_env_id': 'Breakout-v4', 
                          'n_epochs': 6, 
                          'batch_size': 3072, 
                          'n_steps': 128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs': 8,
                          'learning_rate_init': 10e-4,
                          'clip_range_init':    .1,
                          'lr_shedule': 'exponential',
                          'clip_shedule': 'constant',
                          'target_factor': 1e-4
                          })


study1.optimize(instance_objective)