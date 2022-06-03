
# TLDR

[We already set up](../2_baseline) a baseline model which performs better than me. Our goal now is to improve the model to beat the game consistently.

# [3.1_observation_space.py](./3.1_observation_space.py)
Our model gets the image from the game as observation_space - it sees what a human sees. If we use the Atari wrapper it will see a downsampled and trimmed image.


<img src="../pictures/3.1_observation_space_afterWrapper.jpeg" width="400" /> <img src="../pictures/3.1_observation_space_beforeWrapper.jpeg" width="400" />


ATARI wrappper vs original

modifying observation space
trim, grey


# [3.2_aimbot.py](./3.2_aimbot.py)
So now let's build an aimbot - a simple one, that only predicts a straight line. So it can only make a prediction, if the ball travels downwards and is already at a height, where no obstacles can be.
