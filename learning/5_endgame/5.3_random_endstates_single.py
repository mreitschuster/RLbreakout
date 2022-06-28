#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='5.3_random_endstates_single_1'
name_folder='5.3_random_endstates_single'


import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_folder+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_folder+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_folder+'/')
tb_log_name     = name_model



TRAINING_STEPS=1e7

# eval
seed=123
n_envs_eval=8
EVAL_FREQ=10
N_EVAL_EPISODES=10


# base model - the one to continue training
from stable_baselines3 import PPO
#pretrained_model=PPO.load(os.path.expanduser('~/models/breakout-v4/model/3.3_aimbot_training/3.3_aimbot_training_mono_1dim_trim_predict_3fs_0es_seed124_1e7/best_model'))
pretrained_model=None

#%%
from CustomWrapper_random_endstates import wrapper_class_generator, create_env

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
import os


model_params={
            'policy':               'CnnPolicy',
            'n_epochs':              4,
            'batch_size':            256,
            'vf_coef':               0.5,  # trial.suggest_uniform('vf_coef',   0.1, 0.9),
            'ent_coef':              0.01, # trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            'n_steps':               2*64, # trial.suggest_int('n_steps_multiple', 1, 10)*64,
            'learning_rate':         linear_schedule(2.5e-4),
            'clip_range':            linear_schedule(.1),
            #'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            #'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
env_params={
            'env_id'             : 'Breakout-v4',# trial.suggest_categorical('env_id', ['Breakout-v4', 'ALE/Breakout-v5']),  
            'flag_col'           : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
            'flag_dim'           : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
            'flag_predict'       : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
            'frame_stack'        : 3, #trial.suggest_int('frame_stack', 1, 10),
            'MaxAndSkipEnv_skip' : 0, #trial.suggest_int('MaxAndSkipEnv_skip', 0, 4),
            'flag_FireResetEnv'  : True,
            'n_envs'             : 8, #trial.suggest_int('n_envs', 1,16),
            'checkDist'          : 1000, #trial.suggest_int('checkDist', 500,5_000),
            'max_nr_states'      : 100, #trial.suggest_int('max_nr_states', 10,100)
            'prob_start_new'     : 0.3 
            }
       
          
# eval env will be without resampling the endgame and the episodic life
instance_wrapper_class_train = wrapper_class_generator(flag_customObswrapper= True,
                                                               flag_col             = env_params['flag_col'],
                                                               flag_dim             = env_params['flag_dim'],
                                                               flag_predict         = env_params['flag_predict'],
                                                               flag_EpisodicLifeEnv = True,
                                                               flag_FireResetEnv    = env_params['flag_FireResetEnv'],
                                                               MaxAndSkipEnv_skip   = env_params['MaxAndSkipEnv_skip'],
                                                               flag_customEndgameResampler=True,
                                                               checkDist            = env_params['checkDist'],
                                                               max_nr_states        = env_params['max_nr_states'],
                                                               prob_start_new       = env_params['prob_start_new']
                                                               )
        
instance_wrapper_class_eval = wrapper_class_generator(flag_customObswrapper = True,
                                                               flag_col             = env_params['flag_col'],
                                                               flag_dim             = env_params['flag_dim'],
                                                               flag_predict         = env_params['flag_predict'],
                                                               flag_EpisodicLifeEnv = False,
                                                               flag_FireResetEnv    = env_params['flag_FireResetEnv'],
                                                               MaxAndSkipEnv_skip   = env_params['MaxAndSkipEnv_skip'],
                                                               flag_customEndgameResampler=False,
                                                               checkDist            = env_params['checkDist'],
                                                               max_nr_states        = env_params['max_nr_states'],
                                                               prob_start_new       = env_params['prob_start_new']
                                                               )
            
train_env = create_env(env_id=env_params['env_id'], seed=None, wrapper_class=instance_wrapper_class_train, n_envs=env_params['n_envs'], frame_stack=env_params['frame_stack'])
eval_env  = create_env(env_id=env_params['env_id'], seed=None, wrapper_class=instance_wrapper_class_eval,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'])
        
        # eval_freq is the wrong term. it is number of steps after which to evaluate    
        # this is counting in the training environment, which is why we need to adjust for n_envs and not n_envs_eval
        
from stable_baselines3.common.callbacks import EvalCallback
EVAL_STEPS    = int(TRAINING_STEPS/EVAL_FREQ/env_params['n_envs']) 
eval_callback = EvalCallback(eval_env, 
                             best_model_save_path=os.path.join(model_folder,name_model), 
                             n_eval_episodes=N_EVAL_EPISODES, 
                             eval_freq=EVAL_STEPS, 
                             deterministic=False)
                
model = PPO(env=train_env, 
            seed            = seed,
            verbose        = 1, 
            tensorboard_log = tensorboard_folder,
            **model_params) 


if pretrained_model is not None:
    model.set_parameters(pretrained_model.get_parameters())
                
model.learn(total_timesteps = TRAINING_STEPS,
            tb_log_name     = tb_log_name,
            callback        = eval_callback, 
            reset_num_timesteps = False)
            
model.save(os.path.join(model_folder,name_model))


train_env.close()
eval_env.close()   
        
        