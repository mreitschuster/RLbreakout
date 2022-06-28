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
flag_FireResetEnv = False
frame_stack = 3
MaxAndSkipEnv_skip = 0
checkDist = 200
max_nr_states = 100

name_folder='5.1_save_and_load_state'
name_model=name_folder + '_' + str(max_nr_states) + 'states_' + str(checkDist) + 'Checkdist'

import os
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/')
image_folder=os.path.expanduser('~/models/breakout-v4/image/')



# env
env_id                = 'Breakout-v4'
n_envs                = 8


# model
algo                  = 'ppo'
policy                = 'CnnPolicy'
n_steps               = 128
n_epochs              = 4
batch_size            = 256
n_timesteps           = 1e7
learning_rate_initial = 2.5e-4
clip_range_initial    = 0.1
vf_coef               = 0.5
ent_coef              = 0.01

# eval
n_eval_episodes=5
n_eval_envs=1
eval_freq=25000


#%%
import gym
import numpy
import random

class ResampleStatesWrapper(gym.Wrapper):
    def __init__(self, env, checkDist = 10_000, max_nr_states=100):
        super().__init__(env)
        self.checkDist =  checkDist
        self.nextCheckStep = checkDist
        self.states = []
        self.max_nr_states = max_nr_states
        self.n_calls=0
    
    def reset(self, **kwargs):
        if len(self.states)==0:
            return self.env.reset(**kwargs)
        else:
            new_state=random.sample(self.states,1)[0]
            self.env.reset(**kwargs) # to make sure total steps & other variables from the in between wrappers etc are reset
            self.env.restore_state(new_state)   # self.env points to wrapped env
            obs, _, _, _ = self.env.step(self.env.action_space.sample())
            return obs

    
    def step(self, action):
        self.n_calls=self.n_calls+1
        if self.n_calls >= self.nextCheckStep:
            new_state = self.env.clone_state(include_rng=True)
                
            if len(self.states)<self.max_nr_states:
                self.states.append(new_state)
            else:
                i =  random.randrange(0, self.max_nr_states)
                self.states[i] = new_state
                    
                
            self.nextCheckStep = self.nextCheckStep + self.checkDist

        return self.env.step(action)
    
    
#%% we need to recreate the environment, as it is not saved with the model

from BreakoutWrapper import  BreakoutObservationWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper,ClipRewardEnv,EpisodicLifeEnv,MaxAndSkipEnv, FireResetEnv


def wrapper_class_generator(
                  flag_col, 
                  flag_dim, 
                  flag_predict,
                  flag_EpisodicLifeEnv,
                  flag_FireResetEnv,
                  MaxAndSkipEnv_skip,
                  checkDist,
                  max_nr_states):
    
    def wrap_env(env: gym.Env) -> gym.Env:

        env=ResampleStatesWrapper(env, checkDist, max_nr_states)
        if flag_EpisodicLifeEnv:
            env = EpisodicLifeEnv(env)
        if MaxAndSkipEnv_skip>0:
            env=MaxAndSkipEnv(env, skip=MaxAndSkipEnv_skip)
        # think about the order - which wrapper goes when
        env = BreakoutObservationWrapper(env,                     
                                         flag_col    = flag_col, 
                                         flag_dim    = flag_dim, 
                                         flag_predict = flag_predict)
        return(env)

    return wrap_env # we return the function, not the result of the function

instance_wrapper_class=wrapper_class_generator(flag_col    = flag_col,
                                               flag_dim    = flag_dim,
                                               flag_predict = flag_predict,
                                               flag_EpisodicLifeEnv = flag_EpisodicLifeEnv,
                                               flag_FireResetEnv = flag_FireResetEnv,
                                               MaxAndSkipEnv_skip = MaxAndSkipEnv_skip,
                                               checkDist = checkDist,
                                               max_nr_states = max_nr_states)

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


train_env = create_env(env_id=env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack, 
                       wrapper_class=instance_wrapper_class)
eval_env = create_env(env_id=env_id, n_envs=n_eval_envs, seed=seed, frame_stack=frame_stack, 
                       wrapper_class=instance_wrapper_class)


#%%
from stable_baselines3 import PPO
model=PPO.load(os.path.expanduser('~/models/breakout-v4/model/3.3_aimbot_training/3.3_aimbot_training_mono_1dim_trim_predict_3fs_0es_seed124_1e7/best_model'))

    
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
    
    if step==1000:
        pass

    if done.any():
        print('final reward:' + str(cum_reward[done]))
        cum_reward[done]=0
        #eval_env.reset()
        
        
video_recorder.close()
# Close the env
# only this seems to be able to close the window in which the game was rendered
eval_env.close()


