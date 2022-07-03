In this code I tried to run Optuna several times. Either starshaped or stepwise.

# Stepwise
This would mean we first run optuna to find optimal hyperparameters. Then select the best set and train it for a longer period. And then we run another study on the trained model. This was intended as an initial step towards some more continous adaptation of hyperparameters, because I assume the optimal hyperparameters may change with training progress.

![showing score vs timesteps for a growing tree-like training run](../Stepwise.png)



# Star
This would mean we first run optuna to find optimal hyperparameters. Then select the best 5 sets and create a study with them for a longer training period - but again starting from scratch.

One might argue this results in the same as just doing the longer study from the beginning. I see the benefit in being able to be quicker in identify the promising trials for study2. Having some promising trials in the beginning to start with will greatly increase the speed-boost from pruning, as survival probabilities for other trials decrease with cmoing after high-scoring trials.

![showing score vs timesteps for a growing tree-like training run](../Star.png)
