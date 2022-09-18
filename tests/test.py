
from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.game import NoisyBailLeverGame
from noisy_zsc.learner.PPOAgent import PPOAgent
import torch as T
import numpy as np
import matplotlib.pyplot as plt

mean_payoffs = [0., 0., 0.]
bail_payoff = 3.
sigma = 0.5
sigma1 = 0
sigma2 = 0
episode_length = 1


# hyper-parameters
#lr = 0.005
lr = 0.0003
n_epochs = 10 # 3-10
clip = 0.2 # typical 0.1-0.3
N = 20

batch_size = 64


# Initialize environment
env = NoisyBailLeverGame(mean_payoffs, bail_payoff, sigma, sigma1, sigma2, episode_length)
print(env.true_payoffs)
print(env.payoffs1)
print(env.payoffs2)