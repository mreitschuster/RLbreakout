#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

flag_predict = 'predict' # 'nopredict' , 'predict' , 'predict_counters'

n_epochs           = 6
batch_size         = 3072
learning_rate_init = 0.001
clip_range_init    = 0.1
lr_shedule         = 'exponential'
clip_shedule       = 'constant'
target_factor      = 1e-3
n_steps            = 128
TRAINING_STEPS     = 1e7

n_envs=8
name_model='5.1_baseline_v5_tuned_'+str(n_epochs)+'n_epochs_'+str(batch_size)+'batch_size_' + str(n_steps)+'n_steps'+str(learning_rate_init)+lr_shedule + 'lr_' + str(target_factor)+'target_factor_' +str(clip_range_init)+'clip_range_' + str(TRAINING_STEPS)+'steps'
name_folder='5.1_baseline_v5'
study_name=name_model

import os
log_folder         = os.path.expanduser('~/models/breakout-v4/log/'+name_folder+'/')
model_folder       = os.path.expanduser('~/models/breakout-v4/model/'+name_folder+'/')
tensorboard_folder = os.path.expanduser('~/models/breakout-v4/tb_log/'+name_folder+'/')
tb_log_name        = name_model

#%%
import os
from create_objective_shedule import create_objective

#%%
# Creating the first Study 

instance_objective = create_objective(N_EVAL_EPISODES    = 10, 
                                       EVAL_FREQ          = 400, 
                                       TRAINING_STEPS     = TRAINING_STEPS, 
                                       n_envs_eval        = 8,
                                       study_name         = study_name,
                                       model_folder       = model_folder,
                                       tensorboard_folder = tensorboard_folder,
                                       verbose            = 1,
                                       risk_adjustment_stds=1,
                                       N_Rank=int(10/4)    # use lower quartile
                                       )

if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
import yaml
dbconnector = yaml.safe_load(open( os.path.expanduser('~/optunaDB.yaml')))['dbconnector']

import optuna
pruner = optuna.pruners.NopPruner()
study1 = optuna.create_study(direction='maximize', storage=dbconnector, study_name=study_name, load_if_exists=True, pruner=pruner)


study1.enqueue_trial({'train_env_id': 'ALE/Breakout-v5', # 'Breakout-v4', #
                          'n_epochs': n_epochs, 
                          'batch_size': batch_size, 
                          'n_steps':            n_steps, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs':             n_envs,
                          'learning_rate_init': learning_rate_init,
                          'clip_range_init':    clip_range_init,
                          'lr_shedule':         lr_shedule,
                          'clip_shedule':       clip_shedule,
                          'target_factor':      target_factor
                          })
    

study1.enqueue_trial({'train_env_id': 'Breakout-v4', #
                          'n_epochs': n_epochs, 
                          'batch_size': batch_size, 
                          'n_steps':            n_steps, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs':             n_envs,
                          'learning_rate_init': learning_rate_init,
                          'clip_range_init':    clip_range_init,
                          'lr_shedule':         lr_shedule,
                          'clip_shedule':       clip_shedule,
                          'target_factor':      target_factor
                          })
    
# another baseline
study1.enqueue_trial({'train_env_id': 'ALE/Breakout-v5', #
                          'n_epochs': 4, 
                          'batch_size': 256, 
                          'n_steps':            128, 
                          'frame_stack': 4, 
                          'frameskip_env': 4, 
                          'n_envs':             8,
                          'learning_rate_init': 2.5e-4,
                          'clip_range_init':    .1,
                          'lr_shedule':         'linear',
                          'clip_shedule':       'linear',
                          'target_factor':      1
                          })
    
study1.optimize(instance_objective)