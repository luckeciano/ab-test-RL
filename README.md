# ab-test-RL
Using Reinforcement Learning for AB test

![alt text](https://github.com/luckeciano/ab-test-RL/blob/master/teste_ab.png "AB Test")

In this repo, I implemented a code to solve a simple environment of AB test, similar to the [Multi-armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit).

The figure shows the result over time of two "workspaces", A and B. The blue dots is the reward from the actions chosen by the agent. 
The curves, on the other side, refers to the reward curve from each workspace through time.

I used policy gradients implemented in tensorflow. It was my first time coding RL stuff :D
