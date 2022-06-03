
# TLDR

[We already set up](../2_baseline) a baseline model which performs better than me. Our goal now is to improve the model to beat the game consistently.

# [3.1_observation_space.py](./3.1_observation_space.py)
Our model gets the image from the game as observation_space - it sees what a human sees. On the left we see the state after the Atari wrapper - a downsampled, grey image.
On the right we see the original image. When using env.render() we get the original image, I suspect this is because the wrapper hasn't implemented a render() function, so it falls back to the original environment.

<img src="../pictures/3.1_observation_space_beforeWrapper.jpeg" width="400" /><img src="../pictures/3.1_observation_space_afterWrapper.jpeg" width="400" />

So this looks smart - smaller input into our model 84x84x1 instead of 210x160x3, without losing anything we deem necessary for gameplay.


<img src="../pictures/3.1_observation_space_arrays_image.png" width="600" /><img src="../pictures/3.1_observation_space_arrays_state.png" width="600" />

In the arrays we see that the downsampling *smears* the ball a bit. On the original array (right) we see clear cut 200 and 0 values, while on the state after the wrapper we see values in between.


# [3.2_aimbot.py](./3.2_aimbot.py)
So now let's build an aimbot - a simple one, that only predicts a straight line. So it can only make a prediction, if the ball travels downwards and is already at a height, where no obstacles can be.
