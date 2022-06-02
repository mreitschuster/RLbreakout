import os
model_folder=os.path.expanduser('~/models/breakout-v4/')
video_folder=os.path.expanduser('~/models/breakout-v4/video/')
video_file=video_folder+'video.mp4'
#%%

env_id='Breakout-v4'
n_envs        = 1
seed=123

# hyper
frame_stack = 3
flag_plot=False
flag_grey=True
flag_trim=False
prediction_colour=[255,255,255]
prediction_height=3
prediction_width=16
flag_FireResetEnv=False
flag_EpisodicLifeEnv=False
flag_ClipRewardEnv=False
MaxAndSkipEnv_skip=0

#%%

import gym
from breakout_wrapper import Breakout2dObservationWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper,ClipRewardEnv,EpisodicLifeEnv,MaxAndSkipEnv, FireResetEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage

def wrapper_class(env):
    env1 = Breakout2dObservationWrapper(env, 
                                 flag_plot = flag_plot, 
                                 flag_grey = flag_grey, 
                                 flag_trim = flag_trim,
                                 prediction_colour = prediction_colour,
                                 prediction_height = prediction_height,
                                 prediction_width  = prediction_width)
    if flag_FireResetEnv:
        env1 = FireResetEnv(env1)
    if flag_EpisodicLifeEnv:
        env1 = EpisodicLifeEnv(env1)
    if flag_ClipRewardEnv:
        env1 = ClipRewardEnv(env1)
    if MaxAndSkipEnv_skip>0:
        env1=MaxAndSkipEnv(env1, skip=MaxAndSkipEnv_skip)
    
    return(env1)


def create_env(env_id,n_envs=1,seed=123,frame_stack=4):
    new_env=make_vec_env(env_id        = env_id, 
                         n_envs        = n_envs, 
                         seed          = seed,
                         wrapper_class = wrapper_class,   # self.env_wrapper is function get_wrapper_class.<locals>.wrap_env  see line 104 in utils.py
                         vec_env_cls   = DummyVecEnv)    # self.vec_env_class is DummyVecEnv
    
    new_env = VecFrameStack(new_env, frame_stack)  # line 556 in exp_manager.py
    new_env = VecTransposeImage(new_env)           # line 578 in exp_manager.py
    return new_env
    
train_env = create_env(env_id=env_id, n_envs=n_envs, seed=seed, frame_stack=frame_stack)

#%%
from stable_baselines3 import PPO
model = PPO.load(model_folder+'best_model')

#%%
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder
video_recorder = VideoRecorder(train_env, video_file, enabled=True)

train_env.render(mode = "human")
state=train_env.reset()
#train_env.render()

Number_frames= 100000
cumul_reward=0
ep_length=0
for j in range(Number_frames):
    action, _ = model.predict(state)
    #action  = train_env.action_space.sample()
    state, reward, done, info = train_env.step(action)
    train_env.render()
    video_recorder.capture_frame()
    
    cumul_reward=cumul_reward+reward
    ep_length=ep_length+1
    
    image1=state[0,1,:,:]
    #plt.imshow(state[0,0,:,:])
    #plt.show()

    #if done.any():
    if done.any():
        print('cumul_reward:' + str(cumul_reward))
        print('ep_length:' + str(ep_length))
        cumul_reward=0
        ep_length=0
        video_recorder.close()
        video_recorder.enabled = False
        input("Press Enter to continue...")
        train_env.reset()
        break 
        

train_env.close()

#%%

#im = Image.fromarray(arr)