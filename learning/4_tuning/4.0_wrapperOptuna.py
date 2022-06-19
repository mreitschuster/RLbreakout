#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.0_wrapper_optuna'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# env
env_id                = 'Breakout-v4'
n_envs                = 8

# eval
seed=123


flag_col     = 'mono_1dim'        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
flag_dim     = 'trim'        # 'blacken', 'whiten', 'keep', 'trim'
flag_predict = 'predict'   # 'nopredict' , 'predict' , 'predict_counters'
flag_FireResetEnv = False
frame_stack = 3
MaxAndSkipEnv_skip = 0

#%%

import BreakoutWrapper
from BreakoutWrapper import wrapper_class_generator, create_env
    

#%%
from typing import  Callable, Union
def linear_schedule(initial_value: Union[float, str], relative_steps: float) -> Callable[[float], float]:
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return (1 + relative_steps * (progress_remaining-1)) * initial_value

    return func

#%%
from stable_baselines3.common.callbacks import EvalCallback
import gym
import optuna

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
#%%
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

def create_objective(N_EVAL_EPISODES, 
              EVAL_FREQ, 
              TRAINING_STEPS_hp, 
              TRAINING_STEPS, 
              instance_wrapper_class_train, 
              instance_wrapper_class_eval, 
              study_name,
              verbose=0    # 0 no info, 1 starting summary + name per trial, 2 all learning verbosity
              ):
    EVAL_STEPS         = TRAINING_STEPS_hp/EVAL_FREQ/n_envs # eval_freq is the wrong term. it is number of steps after which to evaluate
    relative_steps=TRAINING_STEPS_hp/TRAINING_STEPS
        
    def objective(trial):
        model_params={
            'policy':               'CnnPolicy',
            'n_epochs':              4,
            'batch_size':            256,
            'vf_coef':               trial.suggest_uniform('vf_coef',   0.1, 0.9),
            'ent_coef':              trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            'n_steps':               trial.suggest_int('n_steps_multiple', 1, 10)*64,
          # 'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            'learning_rate': linear_schedule(trial.suggest_loguniform('learning_rate_initial', 1e-8, 1e-1), relative_steps=relative_steps),
            'clip_range':    linear_schedule(trial.suggest_loguniform('clip_range_initial',    1e-4, 0.9 ), relative_steps=relative_steps),
          # 'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
        tb_log_name     = study_name+ "_trial"+str(trial.number)
        
        if verbose>0:
            print(tb_log_name)
            print(trial.params, flush=True)
        
        try:
            train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_train , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
            eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_eval, n_envs=n_envs, seed=seed+1, frame_stack=frame_stack)
            eval_callback = TrialEvalCallback(
                                          eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False
                                         )
            if verbose>1:
                flag_verbose=1
            else:
                flag_verbose=0
                
            model = PPO(env=train_env, 
                         seed            = seed,
                         verbose        = flag_verbose, 
                         tensorboard_log = tensorboard_folder,
                         **model_params) 
            model.learn(total_timesteps = TRAINING_STEPS_hp,
                        tb_log_name     = tb_log_name,
                        callback        = eval_callback)
            SAVE_PATH = os.path.join(model_folder, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)
        except AssertionError as e:
            print(e)
        finally:
            train_env.close()
            eval_env.close()   
        
        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
    
        return eval_callback.last_mean_reward
    return objective
#%%
# Creating the first Study 
instance_wrapper_class_train  =wrapper_class_generator(flag_col    = flag_col,
                                                       flag_dim    = flag_dim,
                                                       flag_predict = flag_predict,
                                                       flag_EpisodicLifeEnv = True,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)
instance_wrapper_class_eval   =wrapper_class_generator(flag_col    = flag_col,
                                                       flag_dim    = flag_dim,
                                                       flag_predict = flag_predict,
                                                       flag_EpisodicLifeEnv = False,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)

instance_objective1 = create_objective(N_EVAL_EPISODES   = 20, 
                                       EVAL_FREQ         = 4, 
                                       TRAINING_STEPS_hp = 1e5, 
                                   TRAINING_STEPS    = 1e6, 
                                   instance_wrapper_class_train=instance_wrapper_class_train, 
                                   instance_wrapper_class_eval =instance_wrapper_class_eval,
                                   study_name="study1",
                                   verbose=1)

study1 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+'4.0_wrapperOptuna_study1.db', study_name='study1', load_if_exists=True)

# adding stable baselines3 zoo hyperparameters
#study1.enqueue_trial({'vf_coef': 0.5,'ent_coef': 0.01, 'n_steps_multiple': 2, 'learning_rate_initial': 2.5e-4, 'clip_range_initial': 0.1 }) # from baselines zoo

# adding some know 5 good runs to make learning a bit faster - especially increasing pruning
#study1.enqueue_trial({'vf_coef': 0.4160655903649405, 'ent_coef': 0.006424932987982504, 'n_steps_multiple': 1, 'learning_rate_initial': 0.000697385903411861, 'clip_range_initial': 0.06579052492108475})
#study1.enqueue_trial({'vf_coef': 0.4359324639911252, 'ent_coef': 0.007611958898675688, 'n_steps_multiple': 3, 'learning_rate_initial': 0.0006339949428944371, 'clip_range_initial': 0.10897847613711822})
#study1.enqueue_trial({'vf_coef': 0.4659268855058361, 'ent_coef': 0.0017325898863186473, 'n_steps_multiple': 1, 'learning_rate_initial': 0.0012380111725726049, 'clip_range_initial': 0.03361172229062028})
#study1.enqueue_trial({'vf_coef': 0.3574396857790585, 'ent_coef': 0.0041725973757283, 'n_steps_multiple': 4, 'learning_rate_initial': 0.00045902088762984515, 'clip_range_initial': 0.22779117371582625})
#study1.enqueue_trial({'vf_coef': 0.41241022396906374, 'ent_coef': 0.0017281826697994676, 'n_steps_multiple': 2, 'learning_rate_initial': 0.0004839355530042937, 'clip_range_initial': 0.11651864434329764})

# some more good ones
#study1.enqueue_trial({'vf_coef': 0.4698571026602645,'ent_coef': 0.0038755980503991745,'n_steps_multiple': 1,'learning_rate_initial': 0.0010127277535755751,'clip_range_initial': 0.12066330358765247})
#study1.enqueue_trial({'vf_coef': 0.4825918289179638,'ent_coef': 0.0015327214515550825,'n_steps_multiple': 1,'learning_rate_initial': 0.025639635755296968,'clip_range_initial': 0.10835916009325867})


import time
time.sleep(3) # otherwise console outputs get mixed up
print('added some known good trials', flush=True)
study1.optimize(instance_objective1, timeout=60*60*3)


#%% 
def get_study_as_sorted_DF(study):
    value=[]
    indexdata=[]
    state=[]
    params=[]
    for i in range(len(study.trials)):
        if (study.trials[i].values is not None):
            value.append(study.trials[i].values[0])
        else:
            value.append(None)
        indexdata.append(study.trials[i].number)
        state.append(study.trials[i].state)
        params.append(study.trials[i].params)
    import pandas
    score_list=pandas.DataFrame({'index':indexdata, 'value':value, 'state':state, 'params':params})
    score_list=score_list.sort_values(by=['value'],ascending=False)
    return score_list

score_list1=get_study_as_sorted_DF(study1)

print(score_list1.state.value_counts())
print(score_list1.iloc[0:5,:])


#%%
instance_objective2 = create_objective(N_EVAL_EPISODES   = 20, 
                                       EVAL_FREQ         = 4, 
                                       TRAINING_STEPS_hp = 1e6, 
                                       TRAINING_STEPS    = 1e7, 
                                       instance_wrapper_class_train=instance_wrapper_class_train, 
                                       instance_wrapper_class_eval =instance_wrapper_class_eval,
                                       study_name="study2",
                                       verbose=1)

study2 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+'4.0_wrapperOptuna_study2.db', study_name='study2', load_if_exists=True)
for i in range(0,5):
    study2.enqueue_trial(score_list1.iloc[i,:].params)
    
    
#y={'vf_coef': 0.4825918289179638,'ent_coef': 0.0015327214515550825,'n_steps_multiple': 1,'learning_rate_initial': 0.025639635755296968,'clip_range_initial': 0.10835916009325867}
#study2.enqueue_trial({'vf_coef': 0.4825918289179638,'ent_coef': 0.0015327214515550825,'n_steps_multiple': 1,'learning_rate_initial': 0.025639635755296968,'clip_range_initial': 0.10835916009325867})

#study2.enqueue_trial(list(score_list1.iloc[0:5,:].params))
study2.optimize(instance_objective2, n_trials=5) # we only want the best 5 trials from previous step



score_list2=get_study_as_sorted_DF(study2)