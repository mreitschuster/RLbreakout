# RLbreakout

The path is designed to help you understand bit by bit. So in the beginning we will focus on simplicity and add upon it step by step.

In my own beginning I started using jupyter notebook, but then switched to a spyder environment, as for me debugging was easier - especially for working through the [stable-baselines3-zoo codebase](https://github.com/DLR-RM/stable-baselines3).


# 1_gym
This is the very basic setup - getting a gym environment to run and train our very first, but not very effective model. More about ensuring we have everything we need - especially a basic understanding.

# 2_baselines
We understand what stable-baselines3-zoo does and reproduce it results with simplified code (for the specific case of PPO on Atari breakout). We take the first steps to optimizing our model, getting one that surpasses my personal capabilities as a casual player:
- use multiple environments in parallel
- adding an evalcallback and a custom callback
- using hyperparameters from stable-baselines3-zoo, including functions for the learning and clip rate

# 3_obswrapper
We add an observation wrapper that removes colour information, trims the picture and adds an Aimbot - additional visual information about where the ball will cut the padel's pane. We then manually tune the framestack hyperparameter and see how the different fetaures of the observation wrapper perform. 

This gets us to a model that gets average rewards of 250 per all lifes, trained with 1e7 timesteps (7 hours on my pc)
https://user-images.githubusercontent.com/41780255/173306620-5c5b1723-987c-46d8-8d3d-935b3c62ea50.mp4


# TODO

explore hyperparameter tuning
resampling late-game states
noise for exploration
other models
on policy/off policy

