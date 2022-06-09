
# Setup
I run everything on a Ubuntu machine. Things should also work on other Linux distributions, Windows or Mac but there will likely be some obstacles you will need to figure out yourself. This setup guide is not intended to cover each single command, but the main steps.

## Conda Environments
I did it using [anaconda](https://www.anaconda.com/). After installing anaconda create a new environment, switch into the environement (activate it), add conda-forge as a repository.

```
conda create --name rl -y
conda activate rl
conda config --add channels conda-forge
conda config --set channel_priority strict
conda update --all
```

If it tells you, it cannot find the command, check that ~/anaconda3/bin/ is in you PATH variable.


From now on whenever we do something on the command line, assure you are in this conda environment. Packages you install inside a environment using conda or pip are only available when you are in the environment.<br>



## IDE
```
conda install spyder==5.3.1 ipdb ipywidgets jupyter_client==7.3.1 ipython==7.33.0 ipykernel==6.13.0 spyder-kernels==2.3.1 python==3.9.13 tensorboard cloudpickle==2.1.0
```

I struggled with the version and took a lot of time to get a working combination of version. Either the debugger is stuck (spyder 5.1.5)or *connecting to kernel* (ipykernel 6.13.1) takes forever, or no ipython shows up.

Please be aware that when you start spyder from your start menu you might not be in the conda environment. So I recommend: open a console: 
```
conda activate rl 
spyder
```

## pytorch on GPU
Go to [pyTorch](https://pytorch.org/get-started/locally/) and do the necessary to get your pytorch running with either your CPU or GPU. For me it was <br>
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
```



## gym & ale & stable

In order to reproduce you need to
```
pip install gym pygame gym[atari] ale-py==0.7.4 stable-baselines3==1.5.0 stable-baselines3[extra] box3d
```

A [gym environment](https://github.com/openai/gym) provides us with a framework to bring games ( environments) together with with models, train them and interact with it. 

ale-py allows us to play Atari games in python.

[stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) provides us with models and wrappers that play the games. 


## Atari Breakout

*find the ROM and put it into a folder*<br>

<code>ale-import-roms /folder_with_rom/</code><br>

In case your console doesn't recognize ale-import-roms, make sure ~.local/bin is in your PATH variable or use the script inside that folder.


## get this repo's code

<code>git clone https://github.com/mreitschuster/RLbreakout.git</code><br>


Now you can proceed to the [first code segments](../1_gym).
