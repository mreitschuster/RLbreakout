#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""
seed=123

import os
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/')
name_model='3.3_aimbot'
image_folder=os.path.expanduser('~/models/breakout-v4/image/')


# env
env_id                = 'Breakout-v4'
n_envs                = 8
frame_stack           = 4

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


#%% new observation wrapper
import gym
import numpy as np
class BreakoutObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, 
                 flag_col='mono_1dim', 
                 flag_dim='trim',
                 flag_predict='predict'
                 ):
        if not(flag_col in ['3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim']):
               raise NameError('unknown value for flag_col')
        if not(flag_dim in ['blacken', 'whiten', 'keep', 'trim']):
               raise NameError('unknown value for flag_dim')
        if not(flag_predict in ['nopredict' , 'predict' ]):
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
        
        if self.flag_predict == 'predict':
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
                        rel_col_prediction=bounces+1+rel_col_prediction
                    
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

def create_env(env_id, n_envs, seed, frame_stack):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         seed          = seed,
                         wrapper_class = BreakoutObservationWrapper,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    
train_env = create_env(env_id=env_id, n_envs=1, seed=seed, frame_stack=frame_stack)


#%% Loading existing baseline model

from stable_baselines3 import PPO
model = PPO(policy, train_env)
baselinemodel=os.path.expanduser('~/models/breakout-v4/model/2.3_copying_hp_zoo/best_model.zip')
assert os.path.exists(baselinemodel) # if it doesnt exist go back to 2.3_copying_hp_zoo.py
model.load(baselinemodel)


#%% Let's see how it plays
import time
import numpy as np
state = train_env.reset()
image=train_env.render(mode='rgb_array')

print(state.shape) # (1,4,84,84)   4 is framestack
print(image.shape) # (210,160,3)   3 is colour channels

#def prep_state(state):
#    image_state=np.stack([state[0,0,:,:],state[0,0,:,:],state[0,0,:,:]],axis=2) # we stack the 1-colour channel 3 times to have a grey image in rgb
#    return image_state

for step in range(int(23)): # we just want some in game pic

    action, _ = model.predict(state)
    state, reward, done, info = train_env.step(action) # state is the picture after wrappers
    image=train_env.render(mode='rgb_array')    # we want tp have access to the image of the underlying environment. 
    
train_env.close()

#%%
from PIL import Image

if state.shape[1]==12: # color + framestack on same dimension
    arr_state=np.transpose(state[0,9:12,:,:],(1,2,0))
elif state.shape[1]==4: # greyscale. 
    arr_state=state[0,3,:,:] # the last of the 4 elements in the second dimension corresponds to current. the others are past.
    if np.amax(arr_state)==1: # monoscale
        arr_state=arr_state*255
else:
    raise NameError("i did not understand the dimensions of the array.")


im1 = Image.fromarray(arr_state) 
im1.save(image_folder+name_model+'_afterWrapper.jpeg')

im2 = Image.fromarray(image)
im2.save(image_folder+name_model+'_beforeWrapper.jpeg')
        
