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
import matplotlib.pyplot as plt

flag_debug=False
flag_viz  =False

class ResampleStatesWrapper(gym.Wrapper):
    def __init__(self, env, checkDist = 500, max_nr_states=100, prob_start_new=0.3):
        super().__init__(env)

        
        # this is the list of states that should be drawn from for training. it is a list
        # of states from before, that have failed checkDist to 2*checkDist steps later
        self.states = []
        self.max_nr_states = max_nr_states   # maximum number of states saved
        self.prob_start_new=prob_start_new   # if not replaying, probability to start new game (other possibility is to draw from self.states)
        
        
        self.n_calls=0
        
        
        # these 2 states are to contionusly save the progress of the current episode
        # (except if it is a replay)
        self.checkDist =  checkDist
        self.nextCheckStep = checkDist
        self.lastEpisode_last_state = None          
        self.lastEpisode_second_last_state = None
        self.lastEpisode_save = True
        
        # replay parameters
        self.replay_failed = 5 # how often should we let the model replay failed states
        self.replay_index  = 0  # counter to count how often we played them. if 0 -> replays finished
        self.replay_state  = None # this is the state we are retrying in a replay 
        self.replay_save   = False
    
    def stop_replay(self):
        # we stop the replay, setting the index back and removing the state object
        
        if self.replay_save:
            if len(self.states)<self.max_nr_states:
                self.states.append(self.lastEpisode_second_last_state)
            else:
                i =  random.randrange(0, self.max_nr_states)
                self.states[i] = self.lastEpisode_second_last_state
        
        self.replay_index  = 0
        self.replay_state  = None
        self.replay_save   = False
        
    def start_new_state(self, **kwargs):
        # start a new state
        # either a completely fresh one or draw from self.states
        
        start_new = (random.uniform(0,1)<self.prob_start_new)
        
        if (len(self.states)==0) or (start_new):
            obs = self.env.reset(**kwargs)
            self.lastEpisode_save = True
            if flag_debug: print("Playing a fresh new game")
            
        else:
            self.lastEpisode_save = False
            new_state=random.sample(self.states,1)[0]
            self.replay_index = 1 # should be played only once and should not be able to spawn new saves
            self.replay_save  = False
            
            if flag_debug: print("loading old gamestate:" + str(new_state))
            self.env.reset(**kwargs) # to make sure total steps & other variables from the in between wrappers etc are reset
            self.env.restore_state(new_state)   # self.env points to wrapped env
            obs, reward, done, info  = self.env.step(self.env.action_space.sample())     
            
            #if done: 
            #    self.stop_replay(save_replay_state_to_list=False)
            assert done==False, "we have loaded a state that fails immediately"
        
        return obs
        
        
    def reset(self, **kwargs):
        
        if self.replay_index == 0: # when we are free to play a new game
            obs = self.start_new_state(**kwargs)
            
            
        else: # when we have to replay a last failed episode
            assert self.replay_state is not None, "replay_state is None, but that should be impossible with replay_index!=0"
            self.env.reset(**kwargs) # to make sure total steps & other variables from the in between wrappers etc are reset
            self.env.restore_state(self.replay_state) 
            
            obs, reward, done, info = self.env.step(self.env.action_space.sample())  
 
            if flag_debug: print("save restored: " + str(self.replay_state))
            if flag_viz and (self.replay_index==self.replay_failed): plt.imshow(obs)    
            if flag_viz and (self.replay_index==self.replay_failed): plt.show()    
 
            
            if flag_debug: print("reset function - resetting to replay with replay_index:" + str(self.replay_index))

            if done:
                # this means the replay_state is not good - it is possible to immediately fail
                self.replay_save   = False
                self.stop_replay() 
                obs = self.start_new_state(**kwargs)
            
        assert (not self.env.needs_reset), "after reset the env should not need resetting"
        
        return obs

    
    def step(self, action):
        self.n_calls=self.n_calls+1
        #if flag_debug: print("step: " + str(self.n_calls))
            
        # Create continous savegames
        # save the last episode in order to have a good start point available
        if self.lastEpisode_save:
            if self.n_calls >= self.nextCheckStep:
                if (self.lastEpisode_last_state is not None) and (not self.env.needs_reset):
                    self.lastEpisode_second_last_state = self.lastEpisode_last_state
                    self.lastEpisode_last_state        = self.env.clone_state(include_rng=True)
                    
                self.nextCheckStep = self.nextCheckStep + self.checkDist
                

        # execute the next action
        obs, reward, done, info = self.env.step(action)
        
        # save last state to main 
        if done: 
            if (self.replay_index !=0):
            # we are anyway replaying an old failure
                if flag_debug: print("replay_index:" + str(self.replay_index) + " return: "+ str(self.env.episode_returns[-1]) + " length: " + str(self.env.episode_lengths[-1]))
                if flag_viz: plt.imshow(obs)
                if flag_viz: plt.show()
                
                # a replay has ended an we need to decrement the index
                self.replay_index = self.replay_index -1
                if self.replay_index ==0: self.stop_replay() 
                # in case the replays were succesfull we can confidently reuse this state for training.
                # what might prohibit this is when the state is such that we cannot get a decent start
                # and immediately fail.
                    
            else:
                if (self.lastEpisode_save == True) and (self.lastEpisode_second_last_state is None):  
                # the model fails very early, before states have been saved
                    if flag_debug: print("failed before having a valid save")            
                elif (self.lastEpisode_save == False):
                    if flag_debug: print("this run was not intended to give a save")  
                else:
 
                # we are not replaying and we have a savegame
                    if flag_debug: print("Episode ended & second last state saved & now starting replays")
                    if flag_debug: print("Orig Gameplay return: "+ str(self.env.episode_returns[-1]) + " length: " + str(self.env.episode_lengths[-1]))
                    if flag_viz: plt.imshow(obs)
                    if flag_viz: plt.show()
                    self.replay_state = self.lastEpisode_second_last_state
                    self.replay_index = self.replay_failed
                    self.replay_save  = True
                        
                    self.lastEpisode_second_last_state = None
                    self.lastEpisode_last_state        = None
                    self.lastEpisode_save              = False # no need to record now, as we replay the just created state
            
            _=1
        
        return obs, reward, done, info 
    
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
    
