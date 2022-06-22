#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=124


flag_col     = 'mono_1dim'        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
flag_dim     = 'trim'        # 'blacken', 'whiten', 'keep', 'trim'
flag_predict = 'predict'   # 'nopredict' , 'predict' 
flag_EpisodicLifeEnv = False # only active for training, not evaluation, otherwise there is confusion when env needs resetting 
flag_FireResetEnv = True
frame_stack = 2
MaxAndSkipEnv_skip = 3

name_model='4.0_wrapper_optuna_PPO'

import os
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
image_folder=os.path.expanduser('~/models/breakout-v4/image/')


# env
env_id                = 'BreakoutNoFrameskip-v4' #'Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5'
n_eval_envs=1

#%%

from stable_baselines3 import PPO
model=PPO.load(model_folder+'model_study2_trial4.zip') # overfitting noFrameskipv4, no Sticky

if False: # do this to compare to baseline
    model=PPO.load('/home/mreit/models/breakout-v4/model/3.3_aimbot_training/3.3_aimbot_training_mono_1dim_trim_predict_3fs_0es_seed124_1e7/best_model')
    frame_stack = 3
    MaxAndSkipEnv_skip = 0

#%%
from BreakoutWrapper import wrapper_class_generator, create_env

instance_wrapper_class=wrapper_class_generator(flag_col    = flag_col,
                                               flag_dim    = flag_dim,
                                               flag_predict = flag_predict,
                                               flag_EpisodicLifeEnv = flag_EpisodicLifeEnv,
                                               flag_FireResetEnv = flag_FireResetEnv,
                                               MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)

eval_env = create_env(env_id=env_id, n_envs=n_eval_envs, seed=seed, frame_stack=frame_stack, 
                       wrapper_class=instance_wrapper_class)



#%% Let's see how it plays

from gym.wrappers.monitoring.video_recorder import VideoRecorder
video_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.mp4')
gif_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.gif')
video_recorder = VideoRecorder(eval_env, video_file, enabled=True)

state=eval_env.reset()
cum_reward=0
for step in range(int(5e3)):
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
cmd1='ffmpeg -i '
cmd2=' -r 10 -f image2pipe -vcodec ppm - | convert -delay 10 -loop 0 -layers Optimize - '
cmd=cmd1+video_file+cmd2+gif_file

os.system(cmd)