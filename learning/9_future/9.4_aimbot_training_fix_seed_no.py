#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=124


flag_col     = 'mono_1dim'        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
flag_dim     = 'trim'        # 'blacken', 'whiten', 'keep', 'trim'
flag_predict = 'predict'   # 'nopredict' , 'predict' , 'predict_counters'
flag_EpisodicLifeEnv = True
flag_FireResetEnv = False
frame_stack = 3
MaxAndSkipEnv_skip = 0
name_folder='9.4_aimbot_training_fix_seed'
name_model=name_folder + '_envseed_notfixed'


import os
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_folder+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_folder+'/'+name_model+'/')
image_folder=os.path.expanduser('~/models/breakout-v4/image/'+name_folder+'/'+name_model+'/')


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


#%% new observation wrapper
import gym
import numpy as np
class BreakoutObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, 
                 flag_col, 
                 flag_dim,
                 flag_predict
                 ):
        if not(flag_col in ['3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim']):
               raise NameError('unknown value for flag_col')
        if not(flag_dim in ['blacken', 'whiten', 'keep', 'trim']):
               raise NameError('unknown value for flag_dim')
        if not(flag_predict in ['nopredict' , 'predict', 'predict_counters' ]):
               raise NameError('unknown value for flag_predict')     
               
        self.prediction_colour=[255,255,255] # painintg the prediction
        self.prediction_height=3
        self.prediction_width=16
        
        self.threshold_ball = 50   # how to recognize the ball
        self.ball_last_col = None  # to store last position
        self.ball_last_row = None    
        
        self.threshold_color=50
        
        self.screen_boundary_left = 8
        self.screen_boundary_right = 151
        
        self.ball_freepane_row_upper = 93
        self.ball_freepane_row_lower = 188
        
        self.padpane_row_upper = 189
        self.padpane_row_lower = 192 
        
        self.pred_pane_row_upper = 200 
        self.pred_pane_row_lower = 205 
        self.pred_pane_width = 6 
        
        self.start_row = 32
        self.end_row   = max(self.pred_pane_row_lower, self.padpane_row_lower)
                
        super().__init__(env)
        
        if flag_dim=='trim':
            rows=self.end_row  - self.start_row
            cols=self.screen_boundary_right-self.screen_boundary_left
        else:
            rows=210
            cols=160
            
        nr_colours=3
        if flag_col=='grey_1dim' or flag_col=='mono_1dim':
            nr_colours=1
                
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(rows, cols,
                                                       nr_colours), 
                                                dtype=np.uint8)
            
        self.flag_col     = flag_col
        self.flag_dim     = flag_dim
        self.flag_predict = flag_predict
    
    def observation(self, obs):
        
        image = obs # we will need to overwrite some pixels
        rel_col_prediction=None
        
        if self.flag_predict in ['predict', 'predict_counters']:
            # ball position
            # we only look into the freepane - the place where the ball can fly unobstructed
            ball_obs = image[self.ball_freepane_row_upper:self.ball_freepane_row_lower, 
                           self.screen_boundary_left:self.screen_boundary_right, 
                           0]
            index_ball=np.where(ball_obs>self.threshold_ball)
            
            if index_ball[0].size>0: # we found the ball
                ball_location_row=np.mean(index_ball[0])+self.ball_freepane_row_upper
                ball_location_col=np.mean(index_ball[1])+self.screen_boundary_left
            else:                    # we have no ball
                ball_location_col=None
                ball_location_row=None
            
            prediction_possible=False # let's decide if we can make a prediction
            if ((ball_location_col is not None)  and 
                (self.ball_last_col is not None) and
                (ball_location_row is not None)  and 
                (self.ball_last_row is not None)):
               
                # ball velocity
                ball_velocity_col = ball_location_col - self.ball_last_col
                ball_velocity_row = ball_location_row - self.ball_last_row
                  
                if (ball_velocity_row>0): 
                    # i cannot make reliable predictions when ball is going up, only when coming down
                    # it could even be 0 if it just got reflected giving us a divison by 0
                    
                    # predicted_impact
                    time_to_impact       = (self.padpane_row_upper-ball_location_row)/ball_velocity_row
                    predicted_impact_col = ball_location_col + time_to_impact*ball_velocity_col
                    
                    
                    col_size             = self.screen_boundary_right-self.screen_boundary_left
                    rel_col_prediction   = (predicted_impact_col-self.screen_boundary_left)/col_size
                                        
                    bounces=int(rel_col_prediction)
                    
                    if np.mod(bounces,2)==0:
                        # even number of bounces -> preserve direction
                        rel_col_prediction=rel_col_prediction-bounces
                    else:
                        rel_col_prediction=bounces+1-rel_col_prediction
                    
                    predicted_impact_col_bounced = rel_col_prediction*col_size + self.screen_boundary_left
                
                    # new image
                    newbar1=round(predicted_impact_col_bounced-self.pred_pane_width/2)
                    newbar2=round(predicted_impact_col_bounced+self.pred_pane_width/2)
                    image[self.pred_pane_row_upper:self.pred_pane_row_lower, newbar1:newbar2,:]=self.prediction_colour
                    prediction_possible=True
                    
                    
                        
            
            if not(prediction_possible):
                predicted_impact_col = None
                predicted_impact_col_bounced = None
                time_to_impact = None
                ball_velocity_col = None
                ball_velocity_row = None
                
                
            # write last ball position
            # important we also do that if we dont make a prediciton!
            self.ball_last_col = ball_location_col
            self.ball_last_row = ball_location_row   
                            
                
        elif self.flag_predict == 'nopredict':
            _ = None
        else:
            raise NameError("flag_predict value unknown")
        

        if self.flag_dim == 'trim':
            image = image[self.start_row:self.end_row, 
                            self.screen_boundary_left:self.screen_boundary_right,
                            :]
            
        elif self.flag_dim == 'keep':
            _ = None
            
        elif self.flag_dim == 'blacken' or self.flag_dim == 'whiten':
            colour=[255,255,255]
            if self.flag_dim == 'blacken':
                colour=[0,0,0]
            image[:self.start_row,:]             = colour
            image[self.end_row:,:]               = colour
            image[:,:self.screen_boundary_left]  = colour
            image[:,self.screen_boundary_right:] = colour
        else:
            raise NameError("flag_dim value unknown")
         
        if self.flag_predict=='predict_counters':
            image[0,0,:]=[0,0,0]
            image[0,1,:]=[0,0,0]
            image[0,2,:]=[0,0,0]
            image[0,3,:]=[0,0,0]
            image[0,4,:]=[0,0,0]
            
            if time_to_impact is not None:
                image[0,0,:]=int(255*min(1,time_to_impact/10))
            if rel_col_prediction is not None:
                image[0,1,:]=int(255*predicted_impact_col_bounced)
            if ball_location_col is not None:
                image[0,2,:]=int(255*min(1,ball_location_col/160))
            if ball_location_row is not None:
                image[0,3,:]=int(255*min(1,ball_location_row/210))
                
            index_pad = np.where(image[self.padpane_row_upper:self.padpane_row_lower,:]>50)
            pad_location_col = np.mean(index_pad[1])
            image[0,4,:]=int(255*min(1,pad_location_col/160))

                
        if self.flag_col=='3col':
            _ = None
        elif self.flag_col == 'grey_3dim':
            image1 = (np.amax(image,2, keepdims=False))
            image  = np.stack([image1,image1,image1],axis=2)
        elif self.flag_col == 'grey_1dim':
            image  = (np.amax(image,2, keepdims=True))
        elif self.flag_col == 'mono_3dim':
            image1 = (np.amax(image,2, keepdims=False)>self.threshold_color)*255.
            image  = np.stack([image1,image1,image1],axis=2)
        elif self.flag_col == 'mono_1dim':
            image  = (np.amax(image,2, keepdims=True)>self.threshold_color)*255.


        else:
            raise NameError("flag_col value unknown")
            

                
        return image


#%% we need to recreate the environment, as it is not saved with the model

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
                  MaxAndSkipEnv_skip):
    
    def wrap_env(env: gym.Env) -> gym.Env:

    
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


def create_env(env_id, 
               wrapper_class,
               n_envs, 
               seed, 
               frame_stack):
    
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                      #   seed          = seed,
                         wrapper_class = wrapper_class,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    
#%%

instance_wrapper_class=wrapper_class_generator(flag_col    = flag_col,
                                               flag_dim    = flag_dim,
                                               flag_predict = flag_predict,
                                               flag_EpisodicLifeEnv = flag_EpisodicLifeEnv,
                                               flag_FireResetEnv = flag_FireResetEnv,
                                               MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)


train_env = create_env(env_id=env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack, 
                       wrapper_class=instance_wrapper_class)

eval_env = create_env(env_id=env_id, n_envs=n_eval_envs, seed=seed, frame_stack=frame_stack, 
                       wrapper_class=instance_wrapper_class)

#%%
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=model_folder,
                             n_eval_episodes=n_eval_episodes,
                           #  log_path=log_folder, 
                             eval_freq=max(eval_freq // n_envs, 1),
                             deterministic=False, 
                             render=False) # see exp_manager.py line 448

#%%
# create learning rate and clip rate functions
# see _preprocess_hyperparams() line 168 in exp_manager.py
# which uses _preprocess_schedules() line 286 in exp_manager.py
# which uses linear_schedule() line 256 in utils.py
from typing import  Callable, Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

learning_rate_shedule = linear_schedule(learning_rate_initial)
clip_range_shedule    = linear_schedule(clip_range_initial)

#%%

from stable_baselines3 import PPO

model = PPO(policy, 
            train_env, 
            n_steps        = n_steps,
            n_epochs       = n_epochs,
            batch_size     = batch_size,
            learning_rate  = learning_rate_shedule,
            clip_range     = clip_range_shedule,
            vf_coef        = vf_coef,
            ent_coef       = ent_coef,            
            verbose        = 1, 
            seed            = seed,
            tensorboard_log = tensorboard_folder) # exp_manager.py line 185

#%%
model.learn(total_timesteps = n_timesteps,
            callback        = eval_callback, 
            tb_log_name     = name_model)

#%%
model.save(model_folder+name_model+'.zip')