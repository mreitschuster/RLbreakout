

We already set up Gym and stable_baselines3 to work in a conda environment with pytorch and tensorboard. We were able to train a PPO model to play breakout. But it doesnt play good.



# 2.1_envs.py
We can use multiple environments at the same time to train in parallel. I removed the piece of code rendering a game. You know how to do it.

For me I felt RAM is the restriction and 12 or more environments would make my system unstable, killing the konsole from which i started the tools and sometimes even logging me out. So I went to 8 envs.

Much better. Instead of an average of 1 reward per episode - 1 block hit before dropping the ball - we get up to 5 with a promising trajectory.


Better. Not great.


# 2.2_eval_callback.py
We want to see evaluation metrics - i.e. using the trained model on a different environment (similar to evaluting on test not training data in non RL). It is necessary that the evaluation environment has the same structure - the same observation_space and action_space, so we need to apply the same wrappers.

Now in the model.learn step gym automatically wraps the (training-)env in a monitor and a VecTransposeImage wrapper. But the eval callback doesn't. After VecTransposeImage the dimensions of the observation space are reordered, so eval_env and env will have different shapes.
```UserWarning: Training and eval env are not of the same type```

So we wrap both manually in VecTransposeImage now. To avoid duplication I wrote a function to create the environment(s).


# 2.3_copying_hp_zoo.py
We havent done any hyperparameter tuning so far. But luckily we can just cheat and copy some known-to-work parameters from [stable-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml).

And more importantly we can learn from the zoo codebase. For that I tend to use the debugger and walk through the code with it. The code 2.3_copying_hp_zoo.py is intended to give you the same result as running zoo on breakout with PPO and all parameters as default.

To ensure this let's run the zoo. Be aware this takes a lot of time, as we aim to do 1e7 timesteps. Running with less timesteps will not provide same results as ctrl-c during training, due to the functional approach to learning rate and clipping, as both scale linearly down to 0 at the last timestep.

```
git clone git@github.com:DLR-RM/rl-baselines3-zoo.git
cd rl-baselines3-zoo
python train.py --algo ppo --env Breakout-v4 --tensorboard-log /models/breakout-v4/tb_log/zoo
```



# remarks

Have a look at your the command line tool nvidia-smi. It tells you how much of you GPU memory is used.If you cannot train a model you might be out of memory. Check here and if necessary close some proceeses. I like having it update continously with the watch command:<br>
<code>watch nvidia-smi</code><br>

Your GPU memory will restrict how many models you can train at the same time.
