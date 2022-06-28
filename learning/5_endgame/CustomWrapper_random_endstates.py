#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""


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


#%%
import random

class ResampleStatesWrapper(gym.Wrapper):
    def __init__(self, env, checkDist = 10_000, max_nr_states=100, prob_start_new=0.3):
        super().__init__(env)
        self.checkDist =  checkDist
        self.nextCheckStep = checkDist
        self.states = []
        self.max_nr_states = max_nr_states
        self.n_calls=0
        self.prob_start_new=prob_start_new
    
    def reset(self, **kwargs):
        
        start_new= (random.uniform(0,1)<self.prob_start_new)
        
        if (len(self.states)==0) or (start_new):
            obs = self.env.reset(**kwargs)
            
        else:
            new_state=random.sample(self.states,1)[0]
            self.env.reset(**kwargs) # to make sure total steps & other variables from the in between wrappers etc are reset
            self.env.restore_state(new_state)   # self.env points to wrapped env
            obs,_,_,_ = self.env.step(self.env.action_space.sample())     
           
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
            
        if self.env.needs_reset:
            _ = self.reset()
       
        x=self.env.step(action)
            
        if len(x)!=4:
            print(x)
            raise NameError("wrong number of elements")
        return x
    
#%% we need to recreate the environment, as it is not saved with the model

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, MaxAndSkipEnv, FireResetEnv


def wrapper_class_generator(
                  flag_customObswrapper,
                  flag_col, 
                  flag_dim, 
                  flag_predict,
                  flag_EpisodicLifeEnv,
                  flag_FireResetEnv,
                  MaxAndSkipEnv_skip,
                  flag_customEndgameResampler,
                  checkDist,
                  max_nr_states,
                  prob_start_new):
    
    def wrap_env(env: gym.Env) -> gym.Env:

        if flag_customEndgameResampler:
            env=ResampleStatesWrapper(env, checkDist, max_nr_states, prob_start_new)
        if flag_EpisodicLifeEnv:
            env = EpisodicLifeEnv(env)
        if MaxAndSkipEnv_skip>0:
            env=MaxAndSkipEnv(env, skip=MaxAndSkipEnv_skip)
        # think about the order - which wrapper goes when
        if flag_customObswrapper:
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
                         seed          = seed,
                         wrapper_class = wrapper_class,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    
