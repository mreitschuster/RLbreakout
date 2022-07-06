#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.2_PPO_grid_1'
study_name=name_model


import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# eval
seed=123


#%%
from BreakoutWrapper import wrapper_class_generator, create_env
from TrialEvalCallback import TrialEvalCallback


#%%
import os
from create_objective import create_objective

#%%
# Creating the first Study 

instance_objective = create_objective(N_EVAL_EPISODES    = 100, 
                                       EVAL_FREQ          = 10, 
                                       TRAINING_STEPS     = 1e7, 
                                       n_envs_eval        = 8,
                                       study_name         = study_name,
                                       model_folder       = model_folder,
                                       tensorboard_folder = tensorboard_folder,
                                       verbose            = 1
                                       
                                       )

if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
import yaml
dbconnector = yaml.safe_load(open( os.path.expanduser('~/optunaDB.yaml')))['dbconnector']

import optuna
study1 = optuna.create_study(direction='maximize', storage=dbconnector, study_name=study_name, load_if_exists=True)


study1.optimize(instance_objective1)

