#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

flag_predict = 'nopredict' # 'nopredict' , 'predict' , 'predict_counters'
n_envs=8
name_model='qrdqn_train_v5__test_NoFrameskip'
name_folder='stochasticity'


import os
log_folder         = os.path.expanduser('~/models/breakout-v4/log/'+name_folder+'/')
model_folder       = os.path.expanduser('~/models/breakout-v4/model/'+name_folder+'/')
tensorboard_folder = os.path.expanduser('~/models/breakout-v4/tb_log/'+name_folder+'/')
tb_log_name        = name_model


TRAINING_STEPS=1e7

# eval
seed=123
n_envs_eval=5
EVAL_FREQ=100
N_EVAL_EPISODES=24

#%%
import yaml
dbconnector = yaml.safe_load(open('optunaDB.yaml'))['dbconnector']

# fill the optunaDB.yaml
# either adjust path or put it into your home directory

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
from typing import  Callable, Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

#%%
from stable_baselines3 import PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.env_util import make_atari_env
import os

def create_objective(N_EVAL_EPISODES, 
              EVAL_FREQ, 
              TRAINING_STEPS, 
              study_name,
              verbose=0    # 0 no info, 1 starting summary + name per trial, 2 all learning verbosity
              ):
       
    def objective(trial):
        model_params={
            'policy':               'CnnPolicy',
            #'n_epochs':              4,
            #'batch_size':            256,
            #'vf_coef':               0.5,  # trial.suggest_uniform('vf_coef',   0.1, 0.9),
            #'ent_coef':              0.01, # trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            #'n_steps':               2*64, # trial.suggest_int('n_steps_multiple', 1, 10)*64,
            #'learning_rate':         linear_schedule(2.5e-4), #trial.suggest_loguniform('learning_rate_initial', 1e-6, 1e-3),
            #'clip_range':            linear_schedule(0.1), # trial.suggest_loguniform('clip_range_initial',    1e-4, 0.9 ),
            'buffer_size':           int(1e6)
        }
        env_params={
            'train_env_id'            : trial.suggest_categorical('env_id', ['Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5']),  
            'eval_env_id'             : trial.suggest_categorical('env_id', ['Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5']),  
            'MaxAndSkipEnv_skip'      : 0, #trial.suggest_int('MaxAndSkipEnv_skip', 0, 4),
            'train_n_envs'            : 8, #trial.suggest_int('n_envs', 1,16)
            'eval_n_envs'             : 8,
            'wrapper_kwargs'          : {'noop_max'              : 30,
                                         'frame_skip'            : 4, 
                                         'screen_size'           : 84, 
                                         'terminal_on_life_loss' : True, 
                                         'clip_reward'           : True
                                         }
            }
        tb_log_name     = study_name+ "_trial"+str(trial.number)
        
        from stable_baselines3.common.env_util import make_atari_env

        train_env = make_atari_env(env_params['train_env_id'], n_envs=env_params['train_n_envs'], wrapper_kwargs=env_params['wrapper_kwargs'])
        eval_env  = make_atari_env(env_params['eval_env_id'],  n_envs=env_params['eval_n_envs'],  wrapper_kwargs=env_params['wrapper_kwargs'])
             
            
        # eval_freq is the wrong term. it is number of steps after which to evaluate    
        # this is counting in the training environment, which is why we need to adjust for n_envs and not n_envs_eval
        
        EVAL_STEPS         = int(TRAINING_STEPS/EVAL_FREQ/env_params['train_n_envs']) 
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False)
        try:
                
            model = QRDQN(env=train_env, 
                         seed            = seed,
                         verbose        = 0, 
                         tensorboard_log = tensorboard_folder,
                         **model_params) 
            
            model.learn(total_timesteps = TRAINING_STEPS,
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
            # Let's not prune
            #raise optuna.exceptions.TrialPruned()
            pass
        return eval_callback.last_mean_reward
    return objective

instance_objective = create_objective(N_EVAL_EPISODES    = 20, 
                                       EVAL_FREQ         = 4, 
                                       TRAINING_STEPS    = 1e7, 
                                       study_name="study2",
                                       verbose=1)

#%%
import optuna
study = optuna.create_study(
                storage=dbconnector,
                load_if_exists=True,
                study_name='stochasticity'
        )

if False:
# make sure to execute this only once, otherwise you add the same trial several times to your DB 
    study.enqueue_trial({'train_env_id':'BreakoutNoFrameskip-v4',
                         'eval_env_id' :'ALE/Breakout-v5'})

    study.enqueue_trial({'train_env_id':'BreakoutNoFrameskip-v4',
                         'eval_env_id' :'Breakout-v4'})
    
    study.enqueue_trial({'train_env_id':'ALE/Breakout-v5',
                         'eval_env_id' :'BreakoutNoFrameskip-v4'})

    study.enqueue_trial({'train_env_id':'Breakout-v4',
                         'eval_env_id' :'BreakoutNoFrameskip-v4'})

study.optimize(instance_objective)


        
        