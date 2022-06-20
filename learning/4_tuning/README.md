
conda install optuna



# using zoo


python train.py --algo ppo --env Breakout-v4 -n 1000000 --seed 123 --tensorboard-log ~/models/breakout-v4/tb_log/zoo -optimize --optimization-log-path ~/models/breakout-v4/opt_log/zoo
