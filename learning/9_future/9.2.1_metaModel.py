#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='9.2.1_metaModel'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# doesnt perform well. maybe a reset button for the underlying model that the agent can use?



# env
#env_id                = 'Breakout-v4'
env_id='BreakoutNoFrameskip-v4'
n_envs                = 8

# eval
seed=123


flag_col     = 'mono_1dim'        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
flag_dim     = 'trim'        # 'blacken', 'whiten', 'keep', 'trim'
flag_predict = 'predict'   # 'nopredict' , 'predict' , 'predict_counters'
flag_FireResetEnv = True
frame_stack = 3
MaxAndSkipEnv_skip = 0

#%%

import BreakoutWrapper
from BreakoutWrapper import wrapper_class_generator, create_env
    

#%%
from stable_baselines3.common.callbacks import EvalCallback
import gym
import optuna

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
#%%
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
 
#%%
# parameters common to all tuning runs

instance_wrapper_class_train  =wrapper_class_generator(flag_col    = flag_col,
                                                       flag_dim    = flag_dim,
                                                       flag_predict = flag_predict,
                                                       flag_EpisodicLifeEnv = True,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)
instance_wrapper_class_eval   =wrapper_class_generator(flag_col    = flag_col,
                                                       flag_dim    = flag_dim,
                                                       flag_predict = flag_predict,
                                                       flag_EpisodicLifeEnv = True,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)

train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_train , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_eval, n_envs=1, seed=seed+1, frame_stack=frame_stack)


#%%
import gym
import numpy
from gym.spaces import Box, Dict, Discrete
from stable_baselines3.common.evaluation import evaluate_policy
class hyperparameter_env(gym.Env):
    def __init__(self, 
                 model_params,
                 train_env,
                 eval_env,
                 tb_log_name,
                 tb_log_folder,
                 riskaversion     = 1, # measured in stdevs
                 #lr_initial       = 1e-4,
                 lr_initial_neg_exp   = 4,
                 eval_episodes    = 3, 
                 max_score        = 1000, 
                 min_score        = 0, 
                 training_steps   = 50000,
                 verbose          = 0
                 ):
        
        super().__init__()
        
        self.lr_exp_scaling      = 0.1
        self.lr_exp              = numpy.array([lr_initial_neg_exp])
        self.reward_abs_last         = 0
        self.eval_episodes       = eval_episodes
        self.recorded_iterations = 10
        self.max_score           = max_score
        
        self.riskaversion = riskaversion
        
        # Specify action space and observation space 
        box_mean_scores = Box(low=min_score, high=max_score, shape=(self.recorded_iterations,), dtype=numpy.double) # np.single=float32
        box_var_scores  = Box(low=0,         high=max_score, shape=(self.recorded_iterations,), dtype=numpy.double) # np.single=float32
        box_lr_exp      = Box(low=-20,         high=20,         shape=(self.recorded_iterations,), dtype=numpy.double) # np.single=float32
        
        # check if we need to provide min and max!?
    
        self.observation_space = Dict({'mean_scores' : box_mean_scores,
                                       'var_scores'  : box_var_scores,
                                       'lr_exp'      : box_lr_exp})
        
        box_action_lr = Box(low=-1, high=1, shape=(1,), dtype=numpy.double) # np.single=float32
        #box_lr = Discrete(low=0, high=15, shape=(1,), dtype=numpy.int8) # np.single=float32
        self.action_space = box_action_lr
        
        # model
        self.train_env      = train_env
        self.eval_env      = eval_env
        self.model = PPO(env             = self.train_env, 
                         seed            = seed,
                         verbose         = (verbose>1), 
                         tensorboard_log = tb_log_folder,
                         learning_rate   = self.get_lr,
                         **model_params) 
        
        self.tb_log_name    = tb_log_name
        self.training_steps = training_steps
        
        # obs
        self.obs = self.observation_space.sample()
        self.obs['mean_scores'] = numpy.zeros((self.recorded_iterations,), dtype=numpy.double)
        self.obs['var_scores']  = numpy.zeros((self.recorded_iterations,), dtype=numpy.double)
        self.obs['lr_exp']      = numpy.zeros((self.recorded_iterations,), dtype=numpy.double)
        
        
        # evalcallback
        
        self.eval_callback = EvalCallback(self.eval_env,
                             #best_model_save_path=model_folder,
                             n_eval_episodes=self.eval_episodes,
                             #log_path=log_folder, 
                             eval_freq=640,
                             #eval_freq=self.training_steps,  #/self.eval_env.n_envs,
                             deterministic=False, 
                             render=False)
        
    def reset(self):
        return self.obs
#        return self.observation_space.sample()
    
    def step(self, action): 
        # Take a step 
        self.lr_exp = self.lr_exp + action * self.lr_exp_scaling
        
        # the learning_rate shedule handed to the model is the get_lr() function 
        #self.logger.record('custom/action', action)
        print('action: ' + str(action)+ '  lr_exp: ' +str(self.lr_exp) + ' lr: ' + str(self.get_lr()))
        self.model.learn(total_timesteps = self.training_steps,
                    tb_log_name          = self.tb_log_name,
                    reset_num_timesteps  = False#, 
                    #callback             = self.eval_callback
                    )
        
        
        mean,std = evaluate_policy(model=self.model, env=self.eval_env, deterministic=False, n_eval_episodes=self.eval_episodes)
        
        #mean,std = evaluate_policy(model=self.model, env=self.eval_env, deterministic=False, n_eval_episodes=3, render=True)
        #mean=self.model.logger.name_to_value['']
        #std=1
        
        reward_abs = mean/self.training_steps + self.riskaversion * std/self.training_steps
        reward = reward_abs - self.reward_abs_last
        self.reward_abs_last = reward_abs
        done = False
        info = {'Nothing' : None}        
        
        # shift all old observations by 1
        self.obs['lr_exp'][0:(self.recorded_iterations-1)]          = self.obs['lr_exp'][1:self.recorded_iterations]
        self.obs['mean_scores'][0:(self.recorded_iterations-1)] = self.obs['mean_scores'][1:self.recorded_iterations]
        self.obs['var_scores'][0:(self.recorded_iterations-1)]  = self.obs['var_scores'][1:self.recorded_iterations]
       
        self.obs['lr_exp'][self.recorded_iterations-1]            = self.lr_exp
        self.obs['mean_scores'][self.recorded_iterations-1] = mean
        self.obs['var_scores'][self.recorded_iterations-1]  = std
        
        return self.obs, reward, done, info
    
    def render(self, *args, **kwargs):
        return self.obs
        
    def close(self):
        pass
    
    def get_lr(self, remainder=1):
        return 10.**(-1.*self.lr_exp[0])
    
    
#%%


hp_env=hyperparameter_env(model_params     = {'policy':           'CnnPolicy',
                                              'n_steps':          128,
                                              'n_epochs':         4,
                                              'batch_size':       256,
                                              'vf_coef':          0.5,
                                              'ent_coef':         0.01,
                                              'clip_range':       0.1 },
                          train_env        = train_env,
                          eval_env         = eval_env,
                          tb_log_name      = 'hp_child',
                          tb_log_folder    = tensorboard_folder,
                          lr_initial_neg_exp       = 4, 
                          eval_episodes    = 3, 
                          max_score        = 1000, 
                          min_score        = 0, 
                          training_steps   = 20000,
                          verbose          = 0)

from stable_baselines3.common import env_checker
env_checker.check_env(hp_env)


#%% 

hp_model_params={'policy':               'MultiInputPolicy',
                'n_steps':               2,
                'n_epochs':              2,
                'batch_size':            2,
                #'vf_coef':               0.5,
                #'ent_coef':              0.01,
                #'learning_rate':         0.001,
                'clip_range':            0.1 }

hp_model_name='hp_model'


hp_model=PPO(env=hp_env, tensorboard_log = tensorboard_folder, **hp_model_params)
#%% 
#load old model
hp_model_old= PPO.load(model_folder+hp_model_name+'.zip')
hp_model.set_parameters(hp_model_old.get_parameters())

#%%
hp_model.learn(total_timesteps = 10000,
            tb_log_name     = hp_model_name)

hp_model.save(model_folder+hp_model_name+'.zip')