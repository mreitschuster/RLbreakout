#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mreitschuster
"""


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
        name = None
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.name      = name
        self.trial     = trial
        self.eval_idx  = 0
        self.is_pruned = False
        self.super     = super()

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            if self.name is not None:
                self.logger.record("eval/"+self.name+"/mean_reward", float(self.last_mean_reward))
            
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
    