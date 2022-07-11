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
study = optuna.create_study(direction='maximize', storage=dbconnector, study_name=study_name, load_if_exists=True)

def set_PRUNED_to_value(study, value):
    trials = []
    for t in study.get_trials():
        if t.state == optuna.trial.TrialState.PRUNED:
            t = optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                params=t.params, 
                distributions=t.distributions,
                user_attrs=t.user_attrs, 
                system_attrs=t.system_attrs,
                intermediate_values=t.intermediate_values)
            trials.append(t)
        elif t.state == optuna.trial.TrialState.COMPLETE:
            trials.append(t)
        else:
            pass
    new_study=optuna.create_study()
    new_study.add_trials(trials)
    return new_study

def subset_study(study, parameter, value_range):
    flag_string=False
    if type(value_range)==list:
        assert len(value_range)==2
    
    elif type(value_range)==int:
        value_range=[value_range,value_range]
        
    elif isinstance(value_range, str):
        flag_string=True
    else:
        raise NameError("value_range not a compatible type")
        
    trials = []
    for t in study.get_trials(): 
        print(t.params)
        if t.params:   # check if not empty
            if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]:
                if flag_string:
                    if t.params[parameter]==value_range:
                        trials.append(t)
                else:
                    if t.params[parameter]>=value_range[0] and t.params[parameter]<=value_range[1]:
                        trials.append(t)
        else:
            pass
    new_study=optuna.create_study()
    new_study.add_trials(trials)
    return new_study


optuna.visualization.matplotlib.plot_param_importances(study)

#all
fig1=optuna.visualization.matplotlib.plot_contour(study, params=["batch_size","n_steps", "n_epochs"])


# batch-size & n_steps at base point
new_study = study
new_study = subset_study(new_study,'n_epochs',4)
new_study = subset_study(new_study,'n_envs',8)
new_study = subset_study(new_study,'frame_stack',4)
new_study = subset_study(new_study,'frameskip_env',4)
new_study = subset_study(new_study,'train_env_id',"Breakout-v4")

fig1=optuna.visualization.matplotlib.plot_contour(new_study, params=["batch_size", "n_steps"])
fig1.title.set_text('epochs=4')
fig1=optuna.visualization.matplotlib.plot_contour(study, params=["batch_size", "n_steps"])
fig1.title.set_text('epochs=4')


# batch-size & epochs
new_study = study
new_study = subset_study(new_study,'n_steps',128)
new_study = subset_study(new_study,'n_envs',8)
new_study = subset_study(new_study,'frame_stack',4)
new_study = subset_study(new_study,'frameskip_env',4)
new_study = subset_study(new_study,'train_env_id',"Breakout-v4")

fig1=optuna.visualization.matplotlib.plot_contour(new_study, params=["batch_size", "n_epochs"])
fig1.title.set_text('n_steps=128')
fig1=optuna.visualization.matplotlib.plot_contour(study, params=["batch_size", "n_epochs"])
fig1.title.set_text('n_steps=128')

# n_steps & epochs at base point
new_study = study
new_study = subset_study(new_study,'batch_size',6*(2**9))
new_study = subset_study(new_study,'n_envs',8)
new_study = subset_study(new_study,'frame_stack',4)
new_study = subset_study(new_study,'frameskip_env',4)
new_study = subset_study(new_study,'train_env_id',"Breakout-v4")

fig1=optuna.visualization.matplotlib.plot_contour(new_study, params=["n_steps", "n_epochs"])
fig1.title.set_text('batch_size=3072')
fig1=optuna.visualization.matplotlib.plot_contour(study, params=["n_steps", "n_epochs"])
fig1.title.set_text('batch_size=3072')

# steps/epochs at 1st optimal  'batch_size': int((2**9)*3), 'n_steps': int((2*64)*3)
new_study = study
new_study = subset_study(new_study,'frame_stack',4)
new_study = subset_study(new_study,'frameskip_env',4)
new_study = subset_study(new_study,'train_env_id',"Breakout-v4")
new_study = subset_study(new_study,'n_envs',8)
new_study = subset_study(new_study,'batch_size', int((2**9)*3))
fig1=optuna.visualization.matplotlib.plot_contour(new_study, params=["n_steps", "n_epochs"])
fig1.title.set_text('batch_size=1536')
fig1=optuna.visualization.matplotlib.plot_contour(study, params=["n_steps", "n_epochs"])
fig1.title.set_text('batch_size=1536')


# batch/epochs at 1st optimal
new_study = study
new_study = subset_study(new_study,'frame_stack',4)
new_study = subset_study(new_study,'frameskip_env',4)
new_study = subset_study(new_study,'train_env_id',"Breakout-v4")
new_study = subset_study(new_study,'n_envs',8)
new_study = subset_study(new_study,'n_steps',    int((2*64)*3))
fig1=optuna.visualization.matplotlib.plot_contour(new_study, params=["batch_size", "n_epochs"])
fig1.title.set_text('n_steps=384')
fig1=optuna.visualization.matplotlib.plot_contour(study, params=["batch_size", "n_epochs"])
fig1.title.set_text('n_steps=384')