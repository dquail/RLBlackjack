#Reinforcement learning for Blackjack

##Problem Description
Consider the task of winning the game of Blackjack. For each possible state of what your cards and the dealers cards, you'd like to know the best action to take. 

##Solution
We will use Reinforcement learning - DoubleQ learning with tabular lookup to find an optimal policy for the player to follow. Doing so will maximize their chances of winning each hand.

##Usage
There are 3 main files:

1. [blackjack.py](blackjack.py)
  * This simulates games of blackjack for our agent to learn from.
2. [DoubleQ.py](DoubleQ.py)
  * Responsible for the agent learning. 
  * Contains a policy - mapping states (dealer cards showing, our cards, usable aces) to optimal actions (hit or stay)
  * learns based on state, reward, and nextState
  
The simplest way to have your agent learn an optimal policy is to have it simulate several games of blackjack. Different results will be achieved with different alpha and epsilon values. And obviously the more simulations are made, the closer to the true value your agent will get. Especially with smaller epsilon and alphas. 

```python
from DoubleQ import *
learn(0.05, 0.01, 500000) #Runs 500,000 simulations using 0.05 alpha and 0.01 epsilon
printPolicy(policy)
```

A grid will displayed for what the agent has learned is the best policy.

##Results
The following are graphs created when running several million episodes with a small alpha and epsilon

![alt text](Policy.png "Optimal actions")

##Reference
This originated from a programming assignment from CMPUT 609 taught by Rich Sutton at the University of Alberta in October of 2016