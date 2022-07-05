#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.1_PPO_env_comparison'

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
from stable_baselines3 import PPO
import os

def create_objective(N_EVAL_EPISODES, 
              EVAL_FREQ, 
              TRAINING_STEPS, 
              n_envs_eval,
              study_name,
              verbose=0    # 0 no info, 1 starting summary + name per trial, 2 all learning verbosity
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
            'n_epochs':              4,
            'batch_size':            256,
            'vf_coef':               0.5,  # trial.suggest_uniform('vf_coef',   0.1, 0.9),
            'ent_coef':              0.01, # trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            'n_steps':               2*64, # trial.suggest_int('n_steps_multiple', 1, 10)*64,
            'learning_rate':         2.5e-4, #trial.suggest_loguniform('learning_rate_initial', 1e-6, 1e-3),
            'clip_range':            0.1, # trial.suggest_loguniform('clip_range_initial',    1e-4, 0.9 ),
            #'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            #'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
        env_params={
            'train_env_id'            : trial.suggest_categorical('train_env_id', ['Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5']),  
            'eval_env_id'             : trial.suggest_categorical('eval_env_id', ['Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5']),  
            # https://www.gymlibrary.ml/environments/atari/breakout/
            'flag_col'     : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
            'flag_dim'     : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
            'flag_predict' : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
            'frame_stack'        : trial.suggest_int('frame_stack', 1, 10),
            'MaxAndSkipEnv_skip' : 0,# trial.suggest_int('MaxAndSkipEnv_skip', 0, 4), # frameskip rather in env
            'flag_FireResetEnv'  : True,
            'n_envs'             : trial.suggest_int('n_envs', 1,16),
            'train_env_kwargs'   : None
            }
        
        frameskip_env = trial.suggest_int('frameskip_env', 1, 8)
        
        env_params['train_env_kwargs'] = env_kwargs(env_params['train_env_id'], frameskip_env)
        env_params['eval_env_kwargs']  = env_kwargs(env_params['eval_env_id'] , frameskip_env)
        
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
        eval_env_v4  = create_env(env_id='Breakout-v4', seed=None, wrapper_class=instance_wrapper_class,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'],            env_kwargs=env_kwargs('Breakout-v4', frameskip_env))
        eval_env_v4NoFS  = create_env(env_id='BreakoutNoFrameskip-v4', seed=None, wrapper_class=instance_wrapper_class,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'], env_kwargs=env_kwargs('BreakoutNoFrameskip-v4', frameskip_env))
        eval_env_v5  = create_env(env_id='ALE/Breakout-v5', seed=None, wrapper_class=instance_wrapper_class,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'],        env_kwargs=env_kwargs('ALE/Breakout-v5', frameskip_env))
        
        # eval_freq is the wrong term. it is number of steps after which to evaluate    
        # this is counting in the training environment, which is why we need to adjust for n_envs and not n_envs_eval
        
        EVAL_STEPS            = int(TRAINING_STEPS/EVAL_FREQ/env_params['n_envs']) 
        eval_callback_v4      = TrialEvalCallback(eval_env_v4,     trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False, name='v4')
        eval_callback_v4NoFS  = TrialEvalCallback(eval_env_v4NoFS, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False, name='v4NoFS')
        eval_callback_v5      = TrialEvalCallback(eval_env_v5,     trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False, name='v5')
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
                        callback        = [eval_callback_v4, eval_callback_v4NoFS, eval_callback_v5])
            SAVE_PATH = os.path.join(model_folder, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)
        except AssertionError as e:
            print(e)
        finally:
            train_env.close()
            eval_callback_v4.close()   
            eval_callback_v4NoFS.close()   
            eval_callback_v5.close()   
        
        #if eval_callback_v5.is_pruned:
        #    raise optuna.exceptions.TrialPruned()
    
        return eval_callback_v5.last_mean_reward
    return objective
#%%
# Creating the first Study 
study_name='4.1_PPO_envs_11'

instance_objective1 = create_objective(N_EVAL_EPISODES   = 100, 
                                       EVAL_FREQ         = 10, 
                                       TRAINING_STEPS    = 1e7, 
                                       n_envs_eval       = 8,
                                       study_name=study_name,
                                       verbose=1)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
import yaml
dbconnector = yaml.safe_load(open( os.path.expanduser('~/optunaDB.yaml')))['dbconnector']

import optuna
study1 = optuna.create_study(direction='maximize', storage=dbconnector, study_name=study_name, load_if_exists=True)


# to this only once
if False:
    study1.enqueue_trial({'train_env_id': 'Breakout-v4',            'frame_stack': 4,'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'BreakoutNoFrameskip-v4', 'frame_stack': 4,'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'ALE/Breakout-v5',        'frame_stack': 4,'frameskip_env': 4, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4',            'frame_stack': 4,'frameskip_env': 6, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'BreakoutNoFrameskip-v4', 'frame_stack': 4,'frameskip_env': 6, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'ALE/Breakout-v5',        'frame_stack': 4,'frameskip_env': 6, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'Breakout-v4',            'frame_stack': 4,'frameskip_env': 8, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'BreakoutNoFrameskip-v4', 'frame_stack': 4,'frameskip_env': 8, 'n_envs': 8}) 
    study1.enqueue_trial({'train_env_id': 'ALE/Breakout-v5',        'frame_stack': 4,'frameskip_env': 8, 'n_envs': 8}) 
# adding known working parameters and expected well-working params

#study1.enqueue_trial({'env_id': 'Breakout-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 0, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'BreakoutNoFrameskip-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 4, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'BreakoutNoFrameskip-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 3, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'BreakoutNoFrameskip-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 5, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'Breakout-v4','frame_stack': 8,'MaxAndSkipEnv_skip': 0, 'n_envs': 7}) 
#study1.enqueue_trial({'env_id': 'ALE/Breakout-v5','frame_stack': 2,'MaxAndSkipEnv_skip': 0, 'n_envs': 1}) 



study1.optimize(instance_objective1)

#%%
optuna.importance.get_param_importances(study1)



