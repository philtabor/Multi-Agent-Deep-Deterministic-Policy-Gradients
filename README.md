# Multi-Agent-Deep-Deterministic-Policy-Gradients
A Pytorch implementation of the multi agent deep deterministic policy gradients(MADDPG) algorithm

This is my implementation of the algorithm presented in the paper: Multi Agent Actor Critic for Mixed Cooperative-Competitive Environments.
You can find this paper here:
https://arxiv.org/pdf/1706.02275.pdf

You will need to install the Multi Agent Particle Environment(MAPE), which you can find here:
https://github.com/openai/multiagent-particle-envs

Make sure to create a virtual environment with the dependencies for the MAPE, since they are somewhat out of date.
I also recommend running this with PyTorch version 1.4.0, as the latest version (1.8) seems to have an issue with
an in place operation I use in the calculation of the critic loss.

It's probably easiest to just clone this repo into the same directory as the MAPE, as the main file requires the
make_env function from that package. 

The video for this tutorial is found here:
https://youtu.be/tZTQ6S9PfkE
