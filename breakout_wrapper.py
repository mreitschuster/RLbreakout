#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:14:08 2022

@author: hendl

https://alexandervandekleut.github.io/gym-wrappers/

https://github.com/openai/gym/tree/master/gym/wrappers


"""
import gym

import numpy as np
import matplotlib.pyplot as plt
import math

class Breakout2dObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, 
                 flag_grey=False, 
                 flag_plot = False, 
                 flag_trim=False, 
                 prediction_colour=[255,255,255],
                 prediction_height=3,
                 prediction_width=16
                 ):
        self.threshold_pad = 50
        self.screen_boundary_left = 8
        self.screen_boundary_right = 151
        self.padpane_row_upper = 189
        self.padpane_row_lower = 192
        
        self.start_row = 32
        self.end_row   = self.padpane_row_lower+prediction_height
        
        self.threshold_ball = 50
        self.ball_freepane_row_upper = 93
        self.ball_freepane_row_lower = 188
        
        self.ball_last_col = None
        self.ball_last_row = None    
        self.prediction_colour=prediction_colour
        self.prediction_height=prediction_height
        self.prediction_width=prediction_width
        
        super().__init__(env)
        
        if flag_trim:
            rows=self.end_row  - self.start_row
            cols=self.screen_boundary_right-self.screen_boundary_left
        else:
            rows=210
            cols=160
            
        nr_colours=3
        if flag_grey:
            nr_colours=1
                
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                    shape=(rows, 
                                                           cols,
                                                           nr_colours), dtype=np.uint8)
            
        self.flag_grey=flag_grey
        self.flag_plot=flag_plot
        self.flag_trim=flag_trim
    
    def observation(self, obs):
        # modify obs
        # obs.shape -> (210, 160, 3)
        # plt.imshow(obs[:,:,0])
        
        # look at obs in variable explorer 
        
        # play area 
        # 32,8 left upper corner
        # 32,151 right upper corner
        
        # blocks 
        # row 62-57 col [200,  72,  72]
        # row 68-63 col [198, 108,  58]
        # row 74-69 col [180, 122,  48]
        # row 80-75 col [162, 162,  42]
        # row 86-81 col [ 72, 160,  72]
        # row 92-87 col [ 66,  72, 200]
        
        # black [0, 0, 0]
        
        # pad's pane : row 189-192 [200,  72,  72]
        
        # ball colour value : [200,  72,  72]
        

        image = obs[:,:,:]
        
        #plt.imshow(image[:,:,1])
        
        # pad position
        pad_obs = obs[self.padpane_row_upper:self.padpane_row_lower, 
                      self.screen_boundary_left:self.screen_boundary_right, 
                      0]
        index_pad=np.where(pad_obs>self.threshold_pad)[1]
        pad_location=np.mean(index_pad)+self.screen_boundary_left
        
        # ball position
        ball_obs = obs[self.ball_freepane_row_upper:self.ball_freepane_row_lower, 
                       self.screen_boundary_left:self.screen_boundary_right, 
                       0]
        index_ball=np.where(ball_obs>self.threshold_ball)
        
        
        if index_ball[0].size>0: # we found the ball
            ball_location_row=np.mean(index_ball[0])+self.ball_freepane_row_upper
            ball_location_col=np.mean(index_ball[1])+self.screen_boundary_left
        else:                    # we have no ball
            ball_location_col=None
            ball_location_row=None
        
        prediction_possible=False
        if ((ball_location_col is not None)  and 
            (self.ball_last_col is not None) and
            (ball_location_row is not None)  and 
            (self.ball_last_row is not None)):
           
            # ball velocity
            ball_velocity_col = ball_location_col - self.ball_last_col
            ball_velocity_row = ball_location_row - self.ball_last_row
              
            if (ball_velocity_row>0):
                # i cannot make reliable predictions when ball is going up, only when coming down
                
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
                    rel_col_prediction=bounces+1+rel_col_prediction
                
                predicted_impact_col_bounced = rel_col_prediction*col_size + self.screen_boundary_left
            
                # new image
                newbar1=round(predicted_impact_col_bounced-8)
                newbar2=round(predicted_impact_col_bounced+8)
                image[self.padpane_row_lower+1:self.padpane_row_lower+3, newbar1:newbar2,:]=self.prediction_colour
                prediction_possible=True
        
        if not(prediction_possible):
            predicted_impact_col = None
            predicted_impact_col_bounced = None
            time_to_impact = None
            ball_velocity_col = None
            ball_velocity_row = None
            
            
        # write last ball position
        self.ball_last_col = ball_location_col
        self.ball_last_row = ball_location_row  

        # trim the image to relevant zones
        if self.flag_trim:
            image_cut = image[self.start_row:self.end_row, 
                              self.screen_boundary_left:self.screen_boundary_right,
                              :]
        else :
            image_cut = image
        
        if self.flag_grey:
            image_cut_Ncol = np.amax(image_cut,2, keepdims=True)
        else:
            image_cut_Ncol = image_cut[:,:,:]
            
        if self.flag_plot:
            plt.imshow(image_cut_Ncol)
            plt.show()     
            
        return image_cut_Ncol
    
    