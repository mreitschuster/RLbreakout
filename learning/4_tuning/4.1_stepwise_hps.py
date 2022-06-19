#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.1_stepwise_hps'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# env
env_id                = 'Breakout-v4'
n_envs                = 8

# eval
seed=123


flag_col     = 'mono_1dim'        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
flag_dim     = 'trim'        # 'blacken', 'whiten', 'keep', 'trim'
flag_predict = 'predict'   # 'nopredict' , 'predict' , 'predict_counters'
flag_FireResetEnv = False
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

def create_objective(N_EVAL_EPISODES, 
              EVAL_FREQ, 
              TRAINING_STEPS, 
              instance_wrapper_class_train, 
              instance_wrapper_class_eval, 
              study_name,
              pretrained_model,
              verbose=0    # 0 no info, 1 starting summary + name per trial, 2 all learning verbosity
              ):
    EVAL_STEPS         = TRAINING_STEPS/EVAL_FREQ/n_envs # eval_freq is the wrong term. it is number of steps after which to evaluate

        
    def objective(trial):
        model_params={
            'policy':               'CnnPolicy',
            'n_epochs':              4,
            'batch_size':            256,
            'vf_coef':               trial.suggest_uniform('vf_coef',   0.1, 0.9),
            'ent_coef':              trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            'n_steps':               trial.suggest_int('n_steps_multiple', 1, 10)*64,
          # 'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            'learning_rate':         trial.suggest_loguniform('learning_rate_initial', 1e-8, 1e-1),
            'clip_range':            trial.suggest_loguniform('clip_range_initial',    1e-4, 0.9 )
          # 'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
        tb_log_name     = study_name+ "_trial"+str(trial.number)
        
        if verbose>0:
            print(tb_log_name)
            print(trial.params, flush=True)
        
        try:
            train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_train , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
            eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_eval, n_envs=n_envs, seed=seed+1, frame_stack=frame_stack)
            eval_callback = TrialEvalCallback(
                                          eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False
                                         )
            if verbose>1:
                flag_verbose=1
            else:
                flag_verbose=0
            
            model = PPO(env=train_env, 
                             seed            = seed,
                             verbose        = flag_verbose, 
                             tensorboard_log = tensorboard_folder,
                             **model_params) 
                
            if pretrained_model is not None:
                model.set_parameters(pretrained_model.get_parameters())
                
            model.learn(total_timesteps     = TRAINING_STEPS,
                        tb_log_name         = tb_log_name,
                        callback            = eval_callback, 
                        reset_num_timesteps = False)
            SAVE_PATH = os.path.join(model_folder, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)
        except AssertionError as e:
            print(e)
        finally:
            train_env.close()
            eval_env.close()   
        
        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()
    
        return eval_callback.last_mean_reward
    return objective

#%%
def train_model(model_params, 
                pretrained_model, 
                N_EVAL_EPISODES, 
                EVAL_FREQ, 
                TRAINING_STEPS, 
                instance_wrapper_class_train, 
                instance_wrapper_class_eval,
                tb_log_name,
                tensorboard_folder,
                verbose=0):
    
    EVAL_STEPS         = TRAINING_STEPS/EVAL_FREQ/n_envs
    
    train_env = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_train , n_envs=n_envs, seed=seed  , frame_stack=frame_stack)
    eval_env  = create_env(env_id=env_id, wrapper_class=instance_wrapper_class_eval, n_envs=n_envs, seed=seed+1, frame_stack=frame_stack)
    
    
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_folder+tb_log_name+'/',
                                 n_eval_episodes=N_EVAL_EPISODES,
                                 eval_freq=EVAL_STEPS,
                                 deterministic=False, 
                                 render=False)
    
    model = PPO(env=train_env, 
                     seed            = seed,
                     verbose        = (verbose>1), 
                     tensorboard_log = tensorboard_folder,
                     **model_params) 
        
    if pretrained_model is not None:
        model.set_parameters(pretrained_model.get_parameters())
        
    model.learn(total_timesteps = TRAINING_STEPS,
                tb_log_name     = tb_log_name,
                callback        = eval_callback, 
                reset_num_timesteps = False)

    return model
    
#%%
def get_study_as_sorted_DF(study):
    value=[]
    indexdata=[]
    state=[]
    params=[]
    for i in range(len(study.trials)):
        if (study.trials[i].values is not None):
            value.append(study.trials[i].values[0])
        else:
            value.append(None)
        indexdata.append(study.trials[i].number)
        state.append(study.trials[i].state)
        params.append(study.trials[i].params)
    import pandas
    score_list=pandas.DataFrame({'index':indexdata, 'value':value, 'state':state, 'params':params})
    score_list=score_list.sort_values(by=['value'],ascending=False)
    return score_list


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
                                                       flag_EpisodicLifeEnv = False,
                                                       flag_FireResetEnv = flag_FireResetEnv,
                                                       MaxAndSkipEnv_skip = MaxAndSkipEnv_skip)

std_objective_params={'N_EVAL_EPISODES' :   20, 
                      'EVAL_FREQ' :          4, 
                      'instance_wrapper_class_train' : instance_wrapper_class_train, 
                      'instance_wrapper_class_eval' :  instance_wrapper_class_eval, 
                      'verbose' :            1}

std_sb3_zoo_hps={'vf_coef': 0.5,
                 'ent_coef': 0.01, 
                 'n_steps_multiple': 2, 
                 'learning_rate_initial': 1e-4,     #2.5e-4 , '# manually reduced as we load near end of training cycle -> lower lr
                 'clip_range_initial': 0.1 }
#%%
# Creating the first Study 

pretrained_model = PPO.load(os.path.expanduser('~/models/breakout-v4/model/3.3_aimbot_training/best_model.zip'))


instance_objective1 = create_objective(TRAINING_STEPS    = 1e5, 
                                       study_name="study1",
                                       pretrained_model=pretrained_model,
                                       **std_objective_params)

study1 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+'4.1_stepwise_hps_study1.db', study_name='study1', load_if_exists=True)
study1.enqueue_trial(std_sb3_zoo_hps) 
study1.optimize(instance_objective1, timeout=60*60*10)


#%% 
score_list1=get_study_as_sorted_DF(study1)

print(score_list1.state.value_counts())
print(score_list1.iloc[0:5,:])

#%%

def create_model_params_from_trial_params(trial_params):
     model_params={
         'policy':               'CnnPolicy',
         'n_epochs':              4,
         'batch_size':            256,
         'vf_coef':               trial_params['vf_coef'],
         'ent_coef':              trial_params['ent_coef'],
         'n_steps':               trial_params['n_steps_multiple']*64,
         'learning_rate':         trial_params['learning_rate_initial'],
         'clip_range':            trial_params['clip_range_initial']
     }
     return model_params

#%%
model_params=create_model_params_from_trial_params(study1.best_params)

pretrained_model2 = train_model(model_params      =model_params, 
                               pretrained_model   =pretrained_model, 
                               TRAINING_STEPS     =3e6, 
                               tb_log_name        ='study1_finalmodel',
                               tensorboard_folder = tensorboard_folder,
                               **std_objective_params)

#pretrained_model3 = PPO.load('/home/mreit/models/breakout-v4/model/4.1_stepwise_hps/study1_finalmodel_step2/best_model')

#%%
instance_objective2 = create_objective(TRAINING_STEPS    = 5e5, 
                                       study_name="study2",
                                       pretrained_model=pretrained_model2,
                                       **std_objective_params)

study2 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+'4.1_stepwise_hps_study2.db', study_name='study2', load_if_exists=True)
study2.enqueue_trial(study1.best_params) 
study2.optimize(instance_objective2, timeout=60*60*10)

#%%
model_params=create_model_params_from_trial_params(study2.best_params)

pretrained_model3 = train_model(model_params      =model_params, 
                               pretrained_model   =pretrained_model2, 
                               TRAINING_STEPS     =3e6, 
                               tb_log_name        ='study1_finalmodel',
                               tensorboard_folder = tensorboard_folder,
                               **std_objective_params)