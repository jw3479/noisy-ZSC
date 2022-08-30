import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from learner.reply_memory import ReplayBuffer
import argparse, os
from copy import deepcopy
import random

from utils.learning_curve import plot_learning_curve


# DQNetwork with checkpoint functionality
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.fc1 = nn.Linear(input_dims, 4)
        self.fc2 = nn.Linear(4, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


# double DQN agent
class DDQNAgent():
    """
    Double Q-Learning agent with fnn function approximation
    Parameters:
        q_net(FNN) - for Q eval and target network
        memory: ReplayBuffer with size mem_size
        batch_size: batch size to sample from ReplayBuffer
        eps_min; eps_dec: epsilon greedy param - setting epsilon decreasing
        replace: update target network with q network every # replace steps.
        algo: string for indicating which algorithm we're using
        env_name: string for indicating which game we're playing
        chkpt_dir: directory for storing model checkpoint
    """
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size,
                 batch_size, eps_min = 0.01, eps_dec = 5e-7, replace = 1000,
                 algo = None, env_name = None, chkpt_dir = 'tmp/dqn'):

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

        self.action_space = [i for i in range(self.n_actions)] # easier to parse actions
        self.learn_step_counter = 0
        # track number of calls to learn to update when to
        # update target network
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = DeepQNetwork(lr, n_actions, "DQNetwork", input_dims, chkpt_dir)

        # target network
        self.q_next = deepcopy(self.q_eval)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # w.p. 1-epsilon, greedy
            # alternative: softmax
            state = T.tensor([observation]).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            # w.p. epsilon, random action
            action = np.random.choice(self.action_space)

        return action


    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()



