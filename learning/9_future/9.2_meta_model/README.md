In this code I tried to create a PPO model that controls the learning rate of the underlying PPO model.
For that I created a gym environment called hyperparameter_env.
Its action space would be increasing or decreasing the learning rate (of the underlying model, not its own) by a multiplicative factor. The observation space is the mean scores, their variance and the learning rate of the last N evaluations (of the underlying model).

I assume I didnt get to good results as trying a meta model would need significantly more training time.



