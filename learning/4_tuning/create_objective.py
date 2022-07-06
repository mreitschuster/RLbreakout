#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""



#%%
from BreakoutWrapper import wrapper_class_generator, create_env
from TrialEvalCallback import TrialEvalCallback


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
            'learning_rate':         2.5e-4, #trial.suggest_loguniform('learning_rate_initial', 1e-6, 1e-3),
            'clip_range':            0.1, # trial.suggest_loguniform('clip_range_initial',    1e-4, 0.9 ),
            #'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            #'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
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

            if verbose>1:
                flag_verbose=1
            else:
                flag_verbose=0
                
            model = PPO(env=train_env, 
                         seed            = seed,
                         verbose        = flag_verbose, 
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
            train_env.close()  
            eval_env_v5.close()   
        
        if eval_callback_v5.is_pruned:
            raise optuna.exceptions.TrialPruned()
    
        return eval_callback_v5.last_mean_reward
    return objective
