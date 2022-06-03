

We already set up Gym and stable_baselines3 to work in a conda environment with pytorch and tensorboard. We were able to train a PPO model to play breakout. But it doesnt play good.



# 2.1_envs.py
We can use multiple environments at the same time to train in parallel. I removed the piece of code rendering a game. You know how to do it.

For me I felt RAM is the restriction and 12 or more environments would make my system unstable, killing the konsole from which i started the tools and sometimes even logging me out. So I went to 8 envs.

Much better. Instead of an average of 1 reward per episode - 1 block hit before dropping the ball - we get up to 5 with a promising trajectory.

Better. Not great.

![2.1v1.3r_ep_rew_mean.png](../pictures/2.1v1.3r_ep_rew_mean.png?raw=true)

# 2.2_callbacks.py
We want to see evaluation metrics - i.e. using the trained model on a different environment (similar to evaluting on test not training data in non RL). It is necessary that the evaluation environment has the same structure - the same observation_space and action_space, so we need to apply the same wrappers.

Now in the model.learn step gym automatically wraps the (training-)env in a monitor and a VecTransposeImage wrapper. But the eval callback doesn't. After VecTransposeImage the dimensions of the observation space are reordered, so eval_env and env will have different shapes.
```UserWarning: Training and eval env are not of the same type```

So we wrap both manually in VecTransposeImage now. To avoid duplication I wrote a function to create the environment(s).


## Record own metrics
In [zoo's enjoy.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/enjoy.py#L179) it states that atari rewards are not necessarily the same as in-game scores. But a [random post on stackoverflow](https://stackoverflow.com/questions/58678710/atari-score-vs-reward-in-rllib-dqn-implementation) states for Breakout it is the same. So let's check that. We add another callback that writes out the supposedely correct score values to our tensorboard (in addition to the already included reward).

And then we find that the
```
_,_,_, infos = env.step(action)
episode_infos = infos[0].get("episode")
episode_infos
```
episode_infos being None. So there (in breakout) is no such variable available. So let us stick with the mean reward.

# [2.3_copying_hp_zoo.py](./2.3_copying_hp_zoo.py)

## baseline zoo
Our aim here is to understand and reproduce a well-performing example from [stable-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/). So let's first downlaod it and run it, so we have results to compare. To ensure this let's run the zoo. Be aware this takes a lot of time, as we aim to do 1e7 timesteps. For me it was half a day. Running with less timesteps will not provide same results as ctrl-c during training, due to the functional approach to learning rate and clipping. You can interupt it with ctrl+c - you will still have the best-model (until that step) and the beginning of the timeseries in tensorboard.

```
git clone git@github.com:DLR-RM/rl-baselines3-zoo.git
cd rl-baselines3-zoo
python train.py --algo ppo --env Breakout-v4 --tensorboard-log ~/models/breakout-v4/tb_log/zoo
```

To understand what the zoo does I used the debugger and walked through the code with it. I made some annotations in my code pointing to the corresponding lines in the zoo code. The code 2.3_copying_hp_zoo.py is intended to give you the same result as running zoo on breakout with PPO and all parameters as default.

![Original Zoo vs 2.3_copying_hp_zoo.py](../pictures/2.3_zoo_vs_copied.png?raw=true)

## Hyperparameters
We havent done any hyperparameter tuning so far. But luckily we can just cheat and copy some known-to-work parameters from [stable-baselines3-zoo's PPO hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).

## Atari Wrapper
Zoo uses the [Atari wrapper](https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html), which includes the EpisodicLifeEnv wrapper we had already used. I recommend reading the docu to better understand which components it has. For our learning curve an in depth knowledge is not required.

## linear_schedule
Learning rate and clip rate are defined as functions, not numbers. Both scale linearly down to 0 at the last timestep. This can be observed in tensorboard.

![2.3_learning_rate.png](../pictures/2.3_learning_rate.png?raw=true)



