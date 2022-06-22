#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.0_wrapper_optuna_PPO'


import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

import optuna
study1 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+name_model+'_study1.db', study_name='study1', load_if_exists=True)

#%%

# '4.0_wrapper_optuna_PPO' used 'env_id' 'frame_stack' 'MaxAndSkipEnv_skip' 'n_envs'

#%%
rend="svg" #"browser"
#optuna.visualization.plot_optimization_history(study1).show(renderer=rend)
#optuna.visualization.plot_intermediate_values(study1).show(renderer=rend)
#optuna.visualization.plot_slice(study1, params=["n_envs", "frame_stack", "MaxAndSkipEnv_skip"]).show(renderer=rend)
#optuna.importance.get_param_importances(study1)
#optuna.visualization.plot_param_importances(study1).show(renderer=rend)

optuna.visualization.plot_parallel_coordinate(study1).show(renderer=rend)
optuna.visualization.plot_contour(study1).show(renderer=rend)

#optuna.visualization.plot_parallel_coordinate(study1, params=["env_id", "frame_stack", "MaxAndSkipEnv_skip"]).show(renderer=rend)
#optuna.visualization.plot_contour(study1, params=["n_envs", "frame_stack", "MaxAndSkipEnv_skip"]).show(renderer=rend)

# show influence on training time
#optuna.visualization.plot_param_importances(
#    study1, target=lambda t: t.duration.total_seconds(), target_name="duration"
#).show(renderer=rend)

#%% adding PRUNED trials to importance analysis
# see https://github.com/optuna/optuna/discussions/3703
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

study_pruned = set_PRUNED_to_value(study1,0)
optuna.visualization.plot_parallel_coordinate(study_pruned).show(renderer=rend)
optuna.visualization.plot_contour(study_pruned).show(renderer=rend)

