
4.1
pip install sb3-contrib


5.5 longtrain
exponential shedule
issue with eval env being stuck and draining fps


python train.py --algo qrdqn --env ALE/Breakout-v5 -n 10000000 --tensorboard-log ~/models/breakout-v4/tb_log/5.1_baseline_qrdqn/qrdqn_zoo_v5

python train.py --algo qrdqn --env Breakout-v4 -n 10000000 --tensorboard-log ~/models/breakout-v4/tb_log/5.1_baseline_qrdqn/qrdqn_zoo_v4

