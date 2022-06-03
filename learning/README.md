# RLbreakout

The path is designed to help you understand bit by bit. So in the beginning we will focus on simplicity and add upon it step by step.

In my own beginning I started using jupyter notebook, but then switched to a spyder environment, as for me debugging was easier - especially for working through the [stable-baselines3-zoo codebase](https://github.com/DLR-RM/stable-baselines3).


# 1_gym

This is the very basic setup - getting a gym environment to run and train our very first, but not very effective model.

# 2_model

We take the first steps to optimizing our model, getting one that surpasses my personal capabilities as a casual player:
- use multiple environments in parallel
- adding a eval callback
- using hyperparameters from stable-baselines3-zoo
- adding a custom observation wrapper


# 3

Here we explore hyperparameter tuning, resampling late-game states and other models.
