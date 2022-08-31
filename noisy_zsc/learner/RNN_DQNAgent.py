# Double Q-Learning to train RNN policies
# Under independent Q-learning: purely treating opponent as environment

import torch.nn as nn # layers
import torch.nn.functional as F #activation function
import torch.optim as optim #optimizers
import torch as T # basic package
from copy import deepcopy
import random

import numpy as np
from .reply_memory import ReplayBuffer
import os




class RNN_DQNetwork(nn.Module):
    def __init__(self, rnn: nn.modules.RNNBase, fnn: nn.Module):
        # set batch_first for the RNN to true
        # batch_first – If True, then the input
        # and output tensors are provided as (batch, seq, feature)
        # instead of (seq, batch, feature).
        super(RNN_DQNetwork, self).__init__()
        self.rnn = rnn
        self.rnn.batch_first = True
        self.fnn = fnn

    def forward(self, input, hidden = None):
        # input shape: (batch_size, seq_length, input_size)
        # rnn_out shape: (batch_size, seq_length, hidden_size)
        # out shape: (batch_size, seq_length, n_actions)
        rnn_out, rnn_hid = self.rnn(input, hidden)
        return self.fnn(rnn_out), rnn_hid # hidden state of RNN

class RNN_DQNAgent:
    """
    Q-learning with RNN function approximation for partially observability
    Parameters:
        q_net(RNN_DQNetwork) - for Q eval and target network
        memory: ReplayBuffer with mem_size
        batch_size: batch size to sample from ReplayBuffer
        eps_min; eps_dec: epsilon greedy param - setting epsilon decreasing
        replace: update target network with q network every # replace steps.
        algo: string for indicating which algorithm we're using
        env_name: string for indicating which game we're playing
        chkpt_dir: directory for storing model checkpoint
    """


    def __init__(self, q_eval: RNN_DQNetwork, gamma, epsilon, lr, n_actions, input_dims, mem_size,
                 batch_size, eps_min = 0.01, eps_dec = 5e-7, replace = 1000,
                 algo = None, env_name = None, chkpt_dir = 'tmp/drqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir

        self.action_space = [i for i in range(self.n_actions)]  # easier to parse actions
        self.learn_step_counter = 0  # track number of calls to learn to update when to update target network

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = q_eval

        self.q_target = deepcopy(q_eval)


        def choose_action(self, observation):
            if np.random.random() > self.epsilon:






