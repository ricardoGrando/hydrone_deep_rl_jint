<h1 align="center">Unmanned Aerial Underwater Vehicle</h1>
<h3 align="center">Double Critic Deep Reinforcement Learning for Mapless Navigation of Unmmaned Aerial Vehicles.</h3>

<p align="center"> 
  <img src="https://img.shields.io/badge/PyTorch-v1.8.1-blue"/>
  <img src="https://img.shields.io/badge/Pandas-v1.2.4-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.20.3-blue"/>
  <img src="https://img.shields.io/badge/Matplotlib-v3.4.2-blue"/>
</p>
<br/>

## Summary
<p align="justify"> 
  <img src="https://i.imgur.com/xNskUKO.png" alt="Hydrone" align="right" width="250">
  <a> We presents a novel deep reinforcement learning-based system for goal-oriented mapless navigation for Unmanned Aerial Vehicles (UAVs). In the context of aerial vehicles, image-based sensing approaches are commonly used, which demands high processing power hardware which can be heavy and difficult to embed into UAVs, mainly for small UAVs. With our proposed approach there is no need for image processing power. Instead, we propose a simple learning system that uses only a few sparse range data from a distance sensor to train the learning agent. We based our approaches on two state-of-art double critic Deep-RL techniques: Twin Delayed Deep Deterministic Policy Gradient (TD3) and Soft Actor-Critic (SAC). We show that our approaches manage to outperform an approach based on the Deep Deterministic Policy Gradient (DDPG) technique and the BUG2 algorithm. Also, our new Deep-RL structure based on Recurrent Neural Networks (RNNs) outperforms the current structure used to perform mapless navigation of mobile robots. Overall, we conclude that Deep-RL approaches based on double critic with Recurrent Neural Networks (RNNs) are better suited to perform mapless navigation and obstacle avoidance of UAVs.</a>  
</p>
  
## Setup
<p align="justify"> 
 <a>All of requirements is show in the badgets above, but if you want to install all of them, enter the repository and execute the following line of code:</a>
</p>

```shell
pip3 install -r requirements.txt
```

<p align="justify"> 
 <a>Before cloning the repository we need to configure your workspace. To do this proceed with the following commands in your terminal:</a>
</p>

```shell
mkdir -p ~/hydrone/src
```
```shell
cd ~/hydrone/
```
```shell
catkin_make
```

<p align="justify"> 
 <a>Now that the workspace is already configured just enter the src folder and clone the repository, finally compile the project. To do this proceed with the following commands in your terminal:</a>
</p>

```shell
cd ~/hydrone/src/
```

```shell
git clone https://github.com/ricardoGrando/hydrone_deep_rl_jint --recursive
```

```shell
cd ~/hydrone/
```

```shell
catkin_make
```

<p align="justify"> 
 <a>We now need to configure your terminal to accept the commands directed to our hydrone workspace. For this you can simply copy the line of code below to your .bashrc (or .zshrc if you use zsh instead of bash) or put the code directly into your terminal. Note that if you choose the second option, every time you open a new terminal you will have to give the following command again.</a>
</p>

<p align="justify"> 
 <a>For <b>bash</b>:</a>
</p>

```shell
source ~/hydrone/devel/setup.bash
```

<p align="justify"> 
 <a>For <b>zsh</b>:</a>
</p>

```shell
source ~/hydrone/devel/setup.zsh
```

<p align="justify"> 
 <a>Okay, now your Hydrone is ready to run!</a><br/>
 <a>To do this, just execute the following command:</a>
</p>

```shell
roslaunch hydrone_deep_rl_jint hydrone.launch
```

## Media

<p align="center"> 
  <img src="media/BUG2.gif" alt="Hydrone Gif" width="400"/>
  <img src="media/DDPG.gif" alt="Hydrone Gif" width="400"/>
  <img src="media/TD3.gif" alt="Hydrone Gif" width="400"/>
  <img src="media/SAC.gif" alt="Hydrone Gif" width="400"/>
</p>  

<p align="justify"> 
  <a>We have the official simulation video posted on youtube, to access it just click on the following hyperlink:</a><a href="https://www.youtube.com/watch?v=Kf5T2SgyiRs"> Video</a>
</p>

<p align="center"> 
  <a><i>If you liked this repository, please don't forget to starred it!</i></a>
  <img src="https://img.shields.io/github/stars/ricardoGrando/hydrone_deep_rl_jint?style=social"/>
</p>




