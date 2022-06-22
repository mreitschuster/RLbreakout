# TLDR

With some tuning of the environment variables we now achieve scores well above 400. The price is that the model doesn't generalize any more and cannot cope with slight stochastic elements in the environment.

**Train RL model to play Breakout - Howto 4/5** Video link - click on the picture.
[![RL Breakout 4/5](../pictures/thumbnails/4_tuning.png)](https://youtu.be/vdtOBbPTGwk)


# Setup
Make sure you have optuna installed:
```
conda install optuna
```
in case you use anaconda. Make sure you execute it in the correct environment.

# [4.0_wrapperOptuna_PPO.py](./4.0_wrapperOptuna_PPO.py)
This code first creates the optuna study *study1* runs it for a few hours (1e6 steps per trial) and then picks the 5 best performing trials and put these in study2, which has a longer traning time (1e7).

# [study_explore.py](./study_explore.py)
This is a script that collects a few lines that allow us to explore the feature importance derived from the tuning results. Optuina by default ignores pruned trials for feature importance. A function has been added to copy the study, set pruned trials to status COMPLETE and give them a score of 0. This can be understood as an imputation of missing values.




