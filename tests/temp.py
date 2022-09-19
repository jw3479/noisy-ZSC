from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.game import NoisyBailLeverGame
from noisy_zsc.learner.PPOAgent import PPOAgent
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import os


# Environment parameters
mean_payoffs = [2., 2., 2.]
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
n_actions = len(env.true_payoffs)
obs_dim = env.obs_dim()
#cur_path = os.getcwd()
actor_chkpt = 'models/actor_'+str(env.sigma1)
critic_chkpt = 'models/critic_'+str(env.sigma1)

agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, 
                    alpha=lr, n_epochs=n_epochs, 
                    input_dims=obs_dim)

agent.load_models(actor_chkpt, critic_chkpt)
