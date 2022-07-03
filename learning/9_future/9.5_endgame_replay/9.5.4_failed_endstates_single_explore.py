#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='5.4_noEndstates_single_5envs_256mb_v5_scratch_1e8_linShedule_128steps'
name_folder='5.4_model_1msteps_performingBadly_score20'


import os
log_folder         = os.path.expanduser('~/models/breakout-v4/log/'+name_folder+'/')
model_folder       = os.path.expanduser('~/models/breakout-v4/model/'+name_folder+'/')
tensorboard_folder = os.path.expanduser('~/models/breakout-v4/tb_log/'+name_folder+'/')
tb_log_name        = name_model



TRAINING_STEPS=1e8

# eval
seed=123
n_envs_eval=1
EVAL_FREQ=40
N_EVAL_EPISODES=10


# base model - the one to continue training
from stable_baselines3 import PPO
pretrained_model=PPO.load(os.path.expanduser('~/models/breakout-v4/model/5.4_failed_endstates_single/best_model_1msteps_performing badly_score20'))
#pretrained_model=None
#%%
from CustomWrapper_failed_endstates import wrapper_class_generator, create_env


#%%
from stable_baselines3 import PPO
import os


env_params={
            'env_id'             : 'ALE/Breakout-v5',# trial.suggest_categorical('env_id', ['Breakout-v4', 'ALE/Breakout-v5']),  
            'flag_col'           : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
            'flag_dim'           : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
            'flag_predict'       : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
            'frame_stack'        : 3, #trial.suggest_int('frame_stack', 1, 10),
            'MaxAndSkipEnv_skip' : 0, #trial.suggest_int('MaxAndSkipEnv_skip', 0, 4),
            'flag_FireResetEnv'  : True,
            'n_envs'             : 5, #trial.suggest_int('n_envs', 1,16),
            'checkDist'          : 200, #trial.suggest_int('checkDist', 500,5_000),
            'max_nr_states'      : 100, #trial.suggest_int('max_nr_states', 10,100)
            'prob_start_new'     : 0.3 
            }
       
instance_wrapper_class_eval = wrapper_class_generator(flag_customObswrapper = True,
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
eval_env  = create_env(env_id=env_params['env_id'], seed=None, wrapper_class=instance_wrapper_class_eval,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'])
        
        # eval_freq is the wrong term. it is number of steps after which to evaluate    
        # this is counting in the training environment, which is why we need to adjust for n_envs and not n_envs_eval
        
#%%

#%% Let's see how it plays

from gym.wrappers.monitoring.video_recorder import VideoRecorder
video_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.mp4')
gif_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.gif')
video_recorder = VideoRecorder(eval_env, video_file, enabled=True)

model=pretrained_model

state=eval_env.reset()
cum_reward=0
for step in range(int(2e4)):
    # do something useful
    action, _ = model.predict(state, deterministic=False)
    state, reward, done, info = eval_env.step(action)
    cum_reward=cum_reward+reward
    image=eval_env.render()
    video_recorder.capture_frame()
    #time.sleep(0.1)


    if done.any():
        print('final reward:' + str(cum_reward[done]))
        cum_reward[done]=0
        #eval_env.reset()
        #break
        
        
video_recorder.close()
# Close the env
# only this seems to be able to close the window in which the game was rendered
eval_env.close()


#%%
        
        