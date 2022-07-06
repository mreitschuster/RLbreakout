#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""

name_model='4.0_wrapper_optuna_PPO'

import os
log_folder=os.path.expanduser('~/models/breakout-v4/log/'+name_model+'/')
model_folder=os.path.expanduser('~/models/breakout-v4/model/'+name_model+'/')
tensorboard_folder=os.path.expanduser('~/models/breakout-v4/tb_log/'+name_model+'/')

# eval
seed=123


#%%
from BreakoutWrapper import wrapper_class_generator, create_env
from TrialEvalCallback import TrialEvalCallback

#%%


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
            'learning_rate':         2.5e-4, #trial.suggest_loguniform('learning_rate_initial', 1e-6, 1e-3),
            'clip_range':            0.1, # trial.suggest_loguniform('clip_range_initial',    1e-4, 0.9 ),
            #'gamma':                 trial.suggest_loguniform('gamma', 0.8, 0.9999),
            #'gae_lambda':            trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        }
        env_params={
            'env_id'       : trial.suggest_categorical('env_id', ['Breakout-v4', 'BreakoutNoFrameskip-v4', 'ALE/Breakout-v5']),  
            # https://www.gymlibrary.ml/environments/atari/breakout/
            'flag_col'     : 'mono_1dim',        # '3col', 'grey_3dim', 'grey_1dim',  'mono_3dim', 'mono_1dim'
            'flag_dim'     : 'trim',       # 'blacken', 'whiten', 'keep', 'trim'
            'flag_predict' : 'predict',  # 'nopredict' , 'predict' , 'predict_counters'
            'frame_stack'        : trial.suggest_int('frame_stack', 1, 10),
            'MaxAndSkipEnv_skip' : trial.suggest_int('MaxAndSkipEnv_skip', 0, 4),
            'flag_FireResetEnv'  : True,
            'n_envs'             : trial.suggest_int('n_envs', 1,16),
            'full_action_space'  :False
            }
        if env_params['env_id']=='Breakout-v4':
            env_kwargs={'full_action_space'         : False,
                        'repeat_action_probability' : 0.,
                        'frameskip'                 : (2,5,)
                        }
            
        elif env_params['env_id']=='BreakoutNoFrameskip-v4':
            env_kwargs={'full_action_space'         : False,
                        'repeat_action_probability' : 0.,
                        'frameskip'                 : 2)
                        }
            
        elif env_params['env_id']=='ALE/Breakout-v5':
            env_kwargs={'full_action_space'         : False,
                        'repeat_action_probability' : 0.25,
                        'frameskip'                 : 2
                        }
        
        tb_log_name     = study_name+ "_trial"+str(trial.number)
        
        if verbose>0:
            print(tb_log_name)
            print(trial.params, flush=True)
        instance_wrapper_class_train  =wrapper_class_generator(flag_col     = env_params['flag_col'],
                                                                   flag_dim     = env_params['flag_dim'],
                                                                   flag_predict = env_params['flag_predict'],
                                                                   flag_EpisodicLifeEnv = True,
                                                                   flag_FireResetEnv = env_params['flag_FireResetEnv'],
                                                                   MaxAndSkipEnv_skip = env_params['MaxAndSkipEnv_skip'])
        instance_wrapper_class_eval   =wrapper_class_generator(flag_col     = env_params['flag_col'],
                                                                   flag_dim     = env_params['flag_dim'],
                                                                   flag_predict = env_params['flag_predict'],
                                                                   flag_EpisodicLifeEnv = False, # will cause "trying to step envioronment that needs reset" otherwise
                                                                   flag_FireResetEnv = env_params['flag_FireResetEnv'],
                                                                   MaxAndSkipEnv_skip = env_params['MaxAndSkipEnv_skip'])
            
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
            model.learn(total_timesteps = TRAINING_STEPS,
                        tb_log_name     = tb_log_name,
                        callback        = eval_callback)
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


instance_objective1 = create_objective(N_EVAL_EPISODES   = 100, 
                                       EVAL_FREQ         = 4, 
                                       TRAINING_STEPS    = 1e6, 
                                       n_envs_eval       = 8,
                                       study_name="study1",
                                       verbose=1)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
study1 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+name_model+'_study1.db', study_name='study1', load_if_exists=True)

# adding known working parameters and expected well-working params

#study1.enqueue_trial({'env_id': 'Breakout-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 0, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'BreakoutNoFrameskip-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 4, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'BreakoutNoFrameskip-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 3, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'BreakoutNoFrameskip-v4','frame_stack': 3,'MaxAndSkipEnv_skip': 5, 'n_envs': 8}) 
#study1.enqueue_trial({'env_id': 'Breakout-v4','frame_stack': 8,'MaxAndSkipEnv_skip': 0, 'n_envs': 7}) 
#study1.enqueue_trial({'env_id': 'ALE/Breakout-v5','frame_stack': 2,'MaxAndSkipEnv_skip': 0, 'n_envs': 1}) 



import time
time.sleep(3) # otherwise console outputs get mixed up
print('starting 1st optimization', flush=True)
study1.optimize(instance_objective1, timeout=60*60*10)

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


#%%
instance_objective2 = create_objective(N_EVAL_EPISODES   = 20, 
                                       EVAL_FREQ         = 4, 
                                       TRAINING_STEPS    = 1e7, 
                                       n_envs_eval           = 8,
                                       study_name="study2",
                                       verbose=1)

study2 = optuna.create_study(direction='maximize', storage='sqlite:///'+model_folder+name_model+'_study2.db', study_name='study2', load_if_exists=True)
for i in range(0,5):
    study2.enqueue_trial(score_list1.iloc[i,:].params)
    
    
#y={'vf_coef': 0.4825918289179638,'ent_coef': 0.0015327214515550825,'n_steps_multiple': 1,'learning_rate_initial': 0.025639635755296968,'clip_range_initial': 0.10835916009325867}
#study2.enqueue_trial({'vf_coef': 0.4825918289179638,'ent_coef': 0.0015327214515550825,'n_steps_multiple': 1,'learning_rate_initial': 0.025639635755296968,'clip_range_initial': 0.10835916009325867})

#study2.enqueue_trial(list(score_list1.iloc[0:5,:].params))
study2.optimize(instance_objective2, n_trials=5) # we only want the best 5 trials from previous step



score_list2=get_study_as_sorted_DF(study2)