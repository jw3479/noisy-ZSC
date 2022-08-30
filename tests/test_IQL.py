import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse, os
from copy import deepcopy
import random

from game import NoisyLeverGame
from learner.DQNLearner import DDQNAgent
from learner.DQNLearner import DeepQNetwork
import matplotlib.pyplot as plt

# testing vanilla independent Q-learning (IQL) with RNN for partial observability
# i.e. each agent treats partner as part of the environment
# without reasoning about joint actions


def convert_dec(myList):
    return list(np.around(np.array(myList),2))

def run():
    game = NoisyLeverGame(
        mean_payoffs=[1., 2., 3.],
        sigma=0,
        sigma1=0.5,
        sigma2=1,
        episode_length=10
    )

    # Setup agents

    agent1 = DDQNAgent(gamma=0.1, epsilon=0.2, lr=0.3, n_actions=len(game.mean_payoffs),
                       input_dims=game.obs_dim(), mem_size=16,
                       batch_size=8, eps_min=0.001,
                       eps_dec=5e-3, replace=1000,
                       algo=None, env_name="noisy_lever", chkpt_dir = 'tmp/dqn')

    agent2 = DDQNAgent(gamma=0.1, epsilon=0.2, lr=0.3, n_actions=len(game.mean_payoffs),
                       input_dims=game.obs_dim(), mem_size=16,
                       batch_size=8, eps_min=0.001,
                       eps_dec=5e-3, replace=1000,
                       algo=None, env_name="noisy_lever", chkpt_dir='tmp/dqn')



    epi_reward_list = []
    for episode in range(10000):
        epi_reward = 0
        print(f'episode: {episode}')
        joint_obs, true_state = game.reset()
        print(f'True world E*: {convert_dec(true_state)}\n')

        obs1 = joint_obs[0]
        obs2 = joint_obs[1]
        print(f'Obs1: {convert_dec(obs1)}, Obs2: {convert_dec(obs2)}\n')

        done = False
        cnt = 0
        while not done:

            obs1 = joint_obs[0]
            obs2 = joint_obs[1]

            if episode % 100 == 0:
                print(agent1.q_eval(T.tensor(obs1)))
            action1 = agent1.choose_action(obs1)
            action2 = agent2.choose_action(obs2)
            ### test against constant action agent ###
            # hard-coded agent 2 who always plays lever 1

            action2 = cnt % 3

            reward, done = game.step(action1, action2)
            epi_reward += reward
            if done:
                epi_reward_list.append(epi_reward)
            print(f'(A1, A2): ({action1}, {action2}) -> Reward: {np.around(reward,2)}, Done: {done}')

            joint_obs_ = game.get_obs()
            obs1_ = joint_obs_[0]
            obs2_ = joint_obs_[1]
            print(f'Obs1: {convert_dec(obs1)}, Obs2: {convert_dec(obs2)}', end='\n' if done else '\n\n')

            # Give experience to learners
            agent1.store_transition(obs1, action1, reward, obs1_, done)
            agent1.store_transition(obs2, action1, reward, obs2_, done)

            # Train learners
            agent1.learn()
            agent2.learn()

            joint_obs = joint_obs_

            cnt += 1
    return epi_reward_list



random.seed(42)
reward_list = run()

plt.plot(reward_list)
plt.show()
