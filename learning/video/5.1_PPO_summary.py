#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123


name_model='5.1_PPO_summary_1'

import os
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
image_folder=os.path.expanduser('~/models/breakout-v4/image/')


# env
env_id                = 'BreakoutNoFrameskip-v4' #'Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5'
n_eval_envs=1

#%%

from stable_baselines3 import PPO
model=PPO.load(model_folder+'trial_10_best_model.zip') # overfitting noFrameskipv4, no Sticky



#%%
env_params={
    'env_id'            : 'Breakout-v4', # 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5' 
    'flag_col'     : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
    'flag_dim'     : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
    'flag_predict' : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
    'frame_stack'        :4,
    'frameskip_env' : 4,
    'MaxAndSkipEnv_skip': 0, # wrapper frameskip
    'flag_FireResetEnv'  : True,
    'n_envs'             : 4,
    'env_kwargs'   : None
    }

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
env_params['env_kwargs'] = env_kwargs(env_params['env_id'], env_params['frameskip_env'])


from BreakoutWrapper_5 import wrapper_class_generator, create_env

instance_wrapper_class  =wrapper_class_generator(flag_col     = env_params['flag_col'],
                                                           flag_dim     = env_params['flag_dim'],
                                                           flag_predict = env_params['flag_predict'],
                                                           flag_EpisodicLifeEnv = True,
                                                           flag_FireResetEnv = env_params['flag_FireResetEnv'],
                                                           MaxAndSkipEnv_skip = env_params['MaxAndSkipEnv_skip'])
eval_env = create_env(env_id=env_params['env_id'], seed=None, wrapper_class=instance_wrapper_class, n_envs=env_params['n_envs'], frame_stack=env_params['frame_stack'], env_kwargs=env_params['env_kwargs'])



#%% Let's see how it plays

from gym.wrappers.monitoring.video_recorder import VideoRecorder
video_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.mp4')
gif_file=os.path.expanduser('~/models/breakout-v4/video/'+name_model+'.gif')
video_recorder = VideoRecorder(eval_env, video_file, enabled=True)

state=eval_env.reset()
cum_reward=0
for step in range(int(5e3)):
    # do something useful
    action, _ = model.predict(state, deterministic=True)
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
#cmd1='ffmpeg -i '
#cmd2=' -r 10 -f image2pipe -vcodec ppm - | convert -delay 10 -loop 0 -layers Optimize - '
#cmd=cmd1+video_file+cmd2+gif_file

#os.system(cmd)