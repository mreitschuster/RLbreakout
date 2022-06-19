#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.0_baseline_optuna'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# env
env_id                = 'Breakout-v4'
n_envs                = 8
frame_stack           = 4

# eval
seed=123


#%%

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper # this includes EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
import gym

def wrapper_class_generator(flag_EpisodicLifeEnv):
    
    def wrap_env(env: gym.Env) -> gym.Env:       
        env = AtariWrapper(env,terminal_on_life_loss=flag_EpisodicLifeEnv)
        return(env)

    return wrap_env # we return the function, not the result of the function



def create_env(env_id, 
               wrapper_class, 
               n_envs, 
               seed, 
               frame_stack):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         seed          = seed,
                         wrapper_class = wrapper_class,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    

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

N_EVAL_EPISODES=20
EVAL_FREQ=5000/n_envs  #
TRAINING_STEPS_hp=1e5
TRAINING_STEPS=1e6
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
    
    instance_wrapper_class_w  =wrapper_class_generator(flag_EpisodicLifeEnv=True)
    instance_wrapper_class_wo =wrapper_class_generator(flag_EpisodicLifeEnv=False)
            

    try:
        train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_w , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
        eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_wo, n_envs=n_envs, seed=seed+1, frame_stack=frame_stack)
        eval_callback = TrialEvalCallback(
                                      eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=False
                                     )
        
        model = PPO(env=train_env, 
                     seed            = seed,
                     verbose        = 1, 
                     tensorboard_log = tensorboard_folder,
                     **model_params) 
        model.learn(total_timesteps = TRAINING_STEPS_hp,
                    callback        = eval_callback)
        SAVE_PATH = os.path.join(model_folder, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
    finally:
        train_env.close()
        eval_env.close()   
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

#%%
# Creating the experiment 

study = optuna.create_study(direction='maximize')

study.enqueue_trial({'vf_coef':               0.5,
                     'ent_coef':              0.01,
                     'n_steps_multiple':         2,
                     'learning_rate_initial': 2.5e-4,
                     'clip_range_initial':    0.1                     
                     }) # from baselines zoo

study.enqueue_trial({'vf_coef':               0.8568474738618462,
                     'ent_coef':               0.008319999374506375,
                     'n_steps_multiple':         2,
                     'learning_rate_initial': 0.0024522885098298814,
                     'clip_range_initial':    0.36422458017134685                     
                     }) # best after hyperparametertuning for 3 hours with 93 trials

#study.optimize(objective, n_trials=10, n_jobs=5)
study.optimize(objective, timeout=60*60*3)

#%%
study.best_params
study.best_trial

#%%
x=[]
for i in range(len(study.trials)):
    x.append(study.trials[i].state)
import pandas
x=pandas.DataFrame(x)
x[0].value_counts()

#%%
instance_wrapper_class_w  =wrapper_class_generator(flag_EpisodicLifeEnv=True)
instance_wrapper_class_wo =wrapper_class_generator(flag_EpisodicLifeEnv=False)

train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_w , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_wo, n_envs=n_envs, seed=seed+1, frame_stack=frame_stack)
eval_callback = EvalCallback(eval_env, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=False)


model_params={
    'policy':               'CnnPolicy',
    'n_epochs':              4,
    'batch_size':            256,
    'vf_coef':               0.8568474738618462,
    'ent_coef':              0.008319999374506375,
    'n_steps':               2*64,
    'learning_rate':  linear_schedule(0.0024522885098298814, relative_steps=1),
    'clip_range':     linear_schedule(0.36422458017134685, relative_steps=1),
}

TRAINING_STEPS=1e7
model = PPO(env=train_env, 
              seed            = seed,
              verbose        = 1, 
              tensorboard_log = tensorboard_folder,
              **model_params) 
model.learn(total_timesteps = TRAINING_STEPS,
             callback        = eval_callback)
SAVE_PATH = os.path.join(model_folder, 'trial_trained_model')
model.save(SAVE_PATH)