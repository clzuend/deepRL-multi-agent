# Project 3 - Collaboration and Competition - Deep Reinforcement Learning Nanodegree

### Introduction

In this project, we use deep reinforcement learning in a multi agent setting to train agents to play tennis in a Unity environment.

<img src="tennis.gif" width="60%" align="top-left" alt="" title="Trained Agent" />

In this environment two agents (tennis rackets) pass a tennis ball. Every time an agent hits the ball over the net, it receives a reward of +0.1. If an agent drops the ball or hits it out of the field, it receives a punishment of -0.01 and the episode ends. The goal is to achieve an average score of at least +0.5 (roughly corresponds to 10 successful passes) over 100 episodes.

Each agent observes 8 variables corresponding to the position and velocity of the ball and racket. However, the Unity environment returns three observations; thus, the actual state size for each agent is 24. The action space consists of two continuous observations: jumping and moving towards or away from the net.


### Setup

#### Download and install deepRL-multi-agent
```
clone git https://github.com/clzuend/deepRL-multi-agent.git
cd deepRL-multi-agent
pip install .
```

#### Download the Unity Tennis Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the project folder. 

3. Depending on your operating system you might have to change the ``TENNIS_PATH`` in  `Multi-Agent-Tennis.ipynb`. 

### Instructions

Open the `Multi-Agent-Tennis.ipynb` workbook to initiate the environment and train the agents.

If one agent object is passed to the training function, the network will be trained jointly for both agents:

```python
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
train()
```

If a second agent object is passed to ``agent2``, the agents will be trained separately:

```python
agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=0)
agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=1)
train(agent1, agent2)
```

If ``competitive = True``, the agents are not explicitly rewarded for hitting the ball over the net, but only for winning the game.

```python
agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=0)
agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=1)
train(agent1, agent2, competitive = True)
```

Additional information can be found in the project report: `Report.pdf`
