#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='5.2_random_endstates_optuna'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# eval
seed=123

# base model - the one to continue training
from stable_baselines3 import PPO
base_model=PPO.load(os.path.expanduser('~/models/breakout-v4/model/3.3_aimbot_training/3.3_aimbot_training_mono_1dim_trim_predict_3fs_0es_seed124_1e7/best_model'))

#%%
from CustomWrapper_random_endstates import wrapper_class_generator, create_env


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
import os

def create_objective(N_EVAL_EPISODES, 
              EVAL_FREQ, 
              TRAINING_STEPS, 
              n_envs_eval,
              study_name,
              pretrained_model,
              verbose=0    # 0 no info, 1 starting summary + name per trial, 2 all learning verbosity
              ):
    
    
    def objective(trial):
        model_params={
            'policy':               'CnnPolicy',
            'n_epochs':              4,
            'batch_size':            256,
            'vf_coef':               0.5,  # trial.suggest_uniform('vf_coef',   0.1, 0.9),
            'ent_coef':              0.01, # trial.suggest_loguniform('ent_coef', 0.0001, 0.9),
            'n_steps':               2*64, # trial.suggest_int('n_steps_multiple', 1, 10)*64,
            'learning_rate':         trial.suggest_loguniform('learning_rate_initial', 1e-6, 1e-4),
            'clip_range':            trial.suggest_loguniform('clip_range_initial',    1e-4, 0.05 ),
            #'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            #'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
        env_params={
            'env_id'             : 'Breakout-v4',# trial.suggest_categorical('env_id', ['Breakout-v4', 'ALE/Breakout-v5']),  
            'flag_col'           : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
            'flag_dim'           : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
            'flag_predict'       : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
            'frame_stack'        : 3, #trial.suggest_int('frame_stack', 1, 10),
            'MaxAndSkipEnv_skip' : 0, #trial.suggest_int('MaxAndSkipEnv_skip', 0, 4),
            'flag_FireResetEnv'  : True,
            'n_envs'             : 8, #trial.suggest_int('n_envs', 1,16),
            'checkDist'          : 5000, #trial.suggest_int('checkDist', 500,5_000),
            'max_nr_states'      : 50, #trial.suggest_int('max_nr_states', 10,100)
            'prob_start_new':            trial.suggest_uniform('prob_start_new',    0.01, 0.9 )
            }
        tb_log_name     = study_name+ "_trial"+str(trial.number)
        
        if verbose>0:
            print(tb_log_name)
            print(trial.params, flush=True)
            
        # eval env will be without resampling the endgame and the episodic life
        instance_wrapper_class_train = wrapper_class_generator(flag_customObswrapper= True,
                                                               flag_col             = env_params['flag_col'],
                                                               flag_dim             = env_params['flag_dim'],
                                                               flag_predict         = env_params['flag_predict'],
                                                               flag_EpisodicLifeEnv = True,
                                                               flag_FireResetEnv    = env_params['flag_FireResetEnv'],
                                                               MaxAndSkipEnv_skip   = env_params['MaxAndSkipEnv_skip'],
                                                               flag_customEndgameResampler=True,
                                                               checkDist            = env_params['checkDist'],
                                                               max_nr_states        = env_params['max_nr_states'],
                                                               prob_start_new       = env_params['prob_start_new']
                                                               )
        
        instance_wrapper_class_eval = wrapper_class_generator(flag_customObswrapper = True,
                                                               flag_col             = env_params['flag_col'],
                                                               flag_dim             = env_params['flag_dim'],
                                                               flag_predict         = env_params['flag_predict'],
                                                               flag_EpisodicLifeEnv = False,
                                                               flag_FireResetEnv    = env_params['flag_FireResetEnv'],
                                                               MaxAndSkipEnv_skip   = env_params['MaxAndSkipEnv_skip'],
                                                               flag_customEndgameResampler=False,
                                                               checkDist            = env_params['checkDist'],
                                                               max_nr_states        = env_params['max_nr_states'],
                                                               prob_start_new       = env_params['prob_start_new']
                                                               )
            
        train_env = create_env(env_id=env_params['env_id'], seed=None, wrapper_class=instance_wrapper_class_train, n_envs=env_params['n_envs'], frame_stack=env_params['frame_stack'])
        eval_env  = create_env(env_id=env_params['env_id'], seed=None, wrapper_class=instance_wrapper_class_eval,  n_envs=n_envs_eval, frame_stack=env_params['frame_stack'])
        
        # eval_freq is the wrong term. it is number of steps after which to evaluate    
        # this is counting in the training environment, which is why we need to adjust for n_envs and not n_envs_eval
        
        EVAL_STEPS         = int(TRAINING_STEPS/EVAL_FREQ/env_params['n_envs']) 
        eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_STEPS, deterministic=False)
        try:

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
                
            model.learn(total_timesteps = TRAINING_STEPS,
                        tb_log_name     = tb_log_name,
                        callback        = eval_callback, 
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
# Creating the first Study 


instance_objective1 = create_objective(N_EVAL_EPISODES   = 20, 
                                       EVAL_FREQ         = 10, 
                                       TRAINING_STEPS    = 2e7, 
                                       n_envs_eval       = 8,
                                       study_name="study1",
                                       pretrained_model  = base_model,
                                       verbose=1)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
study1 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+name_model+'_study1.db', study_name='study1', load_if_exists=True)

# adding known working parameters and expected well-working params

#study1.enqueue_trial({'learning_rate': 8e-6,'clip_range': 4e-3,'prob_start_new': 0.3}) 

import time
time.sleep(3) # otherwise console outputs get mixed up
print('starting 1st optimization', flush=True)
study1.optimize(instance_objective1, timeout=60*60*20)

#%%
optuna.importance.get_param_importances(study1)

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

score_list1=get_study_as_sorted_DF(study1)

print(score_list1.state.value_counts())
print(score_list1.iloc[0:5,:])

