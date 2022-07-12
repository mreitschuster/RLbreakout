#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""



#%%
from BreakoutWrapper import wrapper_class_generator, create_env
from TrialEvalCallback import TrialEvalCallback

#%%
from typing import Callable, Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

import math
def exponential_schedule(initial_value: float, distance: float, target_value_at_end: float) -> Callable[[float], float]:
    
    delta = math.log(initial_value/target_value_at_end)/distance
    
    def func(progress_remaining: float) -> float:
        return (initial_value*math.exp(-1 * delta * (1-progress_remaining) * distance))

    return func

#%%
from stable_baselines3 import PPO
import os
import optuna

def create_objective(N_EVAL_EPISODES, 
              EVAL_FREQ, 
              TRAINING_STEPS, 
              n_envs_eval,
              study_name,
              model_folder,
              tensorboard_folder,
              risk_adjustment_stds,
              N_Rank,
              verbose=0,    # 0 no info, 1 starting summary + name per trial, 2 all learning verbosity
              seed=None
              ):
    
   
    def objective(trial):
        
        
        
        def env_kwargs(env_id, frameskip_env):
            if env_id=='Breakout-v4':
                env_kwargs={'full_action_space'         : False,
                            'repeat_action_probability' : 0.,
                            'frameskip'                 : (frameskip_env-1,frameskip_env+1,)
                            }
                
            elif env_id=='BreakoutNoFrameskip-v4':
                env_kwargs={'full_action_space'         : False,
                            'repeat_action_probability' : 0.,
                            'frameskip'                 : frameskip_env
                            }
                
            elif env_id=='ALE/Breakout-v5':
                env_kwargs={'full_action_space'         : False,
                            'repeat_action_probability' : 0.25,
                            'frameskip'                 : frameskip_env
                            }
            else:
                raise NameError("dont know this env")
                
            return env_kwargs
                
                
        model_params={
            'policy':               'CnnPolicy',
            'n_epochs':              trial.suggest_int('n_epochs', 1, 16), #4
            'batch_size':            trial.suggest_int('batch_size', 32, 4096),  # 256,
            'vf_coef':               0.5,  # trial.suggest_uniform('vf_coef',   0.1, 0.9),
            'ent_coef':              0.01, # trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            'n_steps':               trial.suggest_int('n_steps', 32, 4096),
            'learning_rate_init':    trial.suggest_loguniform('learning_rate_init', 1e-10, 100),
            'clip_range_init':       trial.suggest_loguniform('clip_range_init',    1e-10, 0.9999 ),
            #'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            #'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
            'lr_shedule'            : trial.suggest_categorical('lr_shedule', ['constant', 'linear', 'exponential']),  
            'clip_shedule'            : trial.suggest_categorical('clip_shedule', ['constant', 'linear', 'exponential']),  
            'target_factor':         trial.suggest_loguniform('target_factor', 1e-6,1),
        }
        if model_params['lr_shedule'] == 'constant':
            model_params['learning_rate'] = model_params['learning_rate_init']
        elif model_params['lr_shedule'] == 'linear':
            model_params['learning_rate'] = linear_schedule(model_params['learning_rate_init'])
        elif model_params['lr_shedule'] == 'exponential':
            model_params['learning_rate'] = exponential_schedule(model_params['learning_rate_init'], 1e8, model_params['learning_rate_init']*model_params['target_factor'])
        else: raise NameError("not found")
        
        if model_params['clip_shedule'] == 'constant':
            model_params['clip_range'] = model_params['clip_range_init']
        elif model_params['clip_shedule'] == 'linear':
            model_params['clip_range'] = linear_schedule(model_params['clip_range_init'])
        elif model_params['clip_shedule'] == 'exponential':
            model_params['clip_range'] = exponential_schedule(model_params['clip_range_init'], 1e8, model_params['clip_range_init']* model_params['target_factor'])
        else: raise NameError("not found")
            
        del model_params['learning_rate_init'] 
        del model_params['lr_shedule'] 
        del model_params['clip_range_init'] 
        del model_params['clip_shedule'] 
        del model_params['target_factor'] 
        
        env_params={
            'train_env_id'            : trial.suggest_categorical('train_env_id', ['Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5']),  
            # https://www.gymlibrary.ml/environments/atari/breakout/
            'flag_col'     : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
            'flag_dim'     : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
            'flag_predict' : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
            'frame_stack'        : trial.suggest_int('frame_stack', 1, 10),
            'MaxAndSkipEnv_skip' : 0,# trial.suggest_int('MaxAndSkipEnv_skip', 0, 4), # frameskip rather in env
            'flag_FireResetEnv'  : True,
            'n_envs'             : trial.suggest_int('n_envs', 1, 24),
            'train_env_kwargs'   : None
            }
        
        frameskip_env = trial.suggest_int('frameskip_env', 1, 8)
        
        env_params['train_env_kwargs'] = env_kwargs(env_params['train_env_id'], frameskip_env)
        
        tb_log_name     = study_name+ "_trial"+str(trial.number)
        
        if verbose>0:
            print(tb_log_name)
            print(trial.params, flush=True)
        instance_wrapper_class  =wrapper_class_generator(flag_col     = env_params['flag_col'],
                                                                   flag_dim     = env_params['flag_dim'],
                                                                   flag_predict = env_params['flag_predict'],
                                                                   flag_EpisodicLifeEnv = True,
                                                                   flag_FireResetEnv = env_params['flag_FireResetEnv'],
                                                                   MaxAndSkipEnv_skip = env_params['MaxAndSkipEnv_skip'])
            
        train_env = create_env(env_id=env_params['train_env_id'], seed=None, wrapper_class=instance_wrapper_class, n_envs=env_params['n_envs'], frame_stack=env_params['frame_stack'], env_kwargs=env_params['train_env_kwargs'])
        eval_env_v5  = create_env(env_id='ALE/Breakout-v5', seed=None, wrapper_class=instance_wrapper_class,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'],        env_kwargs=env_kwargs('ALE/Breakout-v5', frameskip_env))
        
        # eval_freq is the wrong term. it is number of steps after which to evaluate    
        # this is counting in the training environment, which is why we need to adjust for n_envs and not n_envs_eval
        
        EVAL_STEPS            = int(TRAINING_STEPS/EVAL_FREQ/env_params['n_envs']) 
        eval_callback_v5      = TrialEvalCallback(eval_env_v5,     trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False, risk_adjustment_stds=risk_adjustment_stds,N_Rank=N_Rank)
        try:
               
            model = PPO(env=train_env, 
                         seed            = seed,
                         verbose        = (verbose>=1), 
                         tensorboard_log = tensorboard_folder,
                         **model_params) 
            model.learn(total_timesteps = TRAINING_STEPS,
                        tb_log_name     = tb_log_name,
                        callback        = eval_callback_v5)
            SAVE_PATH = os.path.join(model_folder, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)
        except AssertionError as e:
            print(e)
        finally:
            try: train_env.close()  
            except: train_env = None
        
            try: eval_env_v5.close()  
            except: eval_env_v5 = None
                
        if eval_callback_v5.is_pruned:
            raise optuna.exceptions.TrialPruned()
    
        #return eval_callback_v5.last_mean_reward
        return eval_callback_v5.last_median_reward
    return objective
