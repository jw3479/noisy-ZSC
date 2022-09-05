# Double Q-Learning to train RNN policies
# Under independent Q-learning: purely treating opponent as environment

from itertools import zip_longest
from typing import Iterable, Optional
import torch.nn as nn # layers
import torch.nn.functional as F #activation function
import torch.optim as optim #optimizers
import torch as T # basic package
from copy import deepcopy
import random

import numpy as np
from .reply_memory import EpisodeBuffer, ReplayBuffer
import os


## Matthias' polyak update
def zip_strict(*iterables: Iterable) -> Iterable:
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[nn.Parameter], ## param of current network
    target_params: Iterable[nn.Parameter],  ## param of current network
    tau: float,
) -> None:
    """
    Polyak (soft) target network update.
    Inspired by implementation of stable-baselines3:
    https://github.com/DLR-RM/stable-baselines3/
    """
    with T.no_grad():
        # Use `zip_strict`, since `zip` does not raise an excception if
        # lengths of parameter iterables differ.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau) # 1-tau of target # tau of online
            T.add(
                target_param.data, param.data, alpha=tau, out=target_param.data)


class DDRQNetwork(nn.Module):
    """
    DDRQNetwork: network architecture for online and target network in double Q-learning
    First pass though lstm layer for action observation history
    then to feedforward nn
    """
    def __init__(self, lr, n_actions, input_dims, hidden_units):
        super(DDRQNetwork, self).__init__()
    
        n_hidden_units = hidden_units

        # first pass through lstm layer
        self.lstm = nn.LSTM(input_dims + n_actions, hidden_size = n_hidden_units, num_layers = 1, bias = True, batch_first = True)

        self.fc1 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, hidden=None):
        x, hidden = self.lstm(state, hidden)
        x = F.relu(self.fc1(x))
        return self.fc2(x), hidden


class DDRQNAgent:
    """
    DDRQN: Double Q-learning with RNN function approximation for partially observability
    Parameters:
        q_net(DDRQNetwork) - for Q eval and target network
        memory: ReplayBuffer with mem_size
        batch_size: batch size to sample from ReplayBuffer
        eps_min; eps_dec: epsilon greedy param - setting epsilon decreasing
        replace: update target network with q network every # replace steps.
        algo: string for indicating which algorithm we're using
        env_name: string for indicating which game we're playing
        chkpt_dir: directory for storing model checkpoint
    """

    def __init__(self, q_eval: DDRQNetwork, gamma, epsilon, n_actions, input_dims, mem_size,
                 batch_size, episode_length, eps_min = 0.01, eps_dec = 5e-5, replace = 1000,
                 tau = 1.0):

        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        
        self.tau = tau
        
        self.action_space = [i for i in range(self.n_actions)] # easier to parse actions
        self.learn_step_counter = 0
        # track number of calls to learn to update when to
        # update target network
        self.memory = EpisodeBuffer(mem_size, input_dims, episode_length)

        self.q_eval = q_eval

        # target network
        self.q_next = deepcopy(self.q_eval)
        self.hidden = None

    def choose_action(self, observation, last_action: Optional[int], epsilon = None):
        if epsilon is None:
            epsilon = self.epsilon
        
        a = T.zeros((self.n_actions)) if last_action is None else T.nn.functional.one_hot(T.tensor(last_action), self.n_actions)
        state = T.concat((a, T.tensor(list(observation)).to(self.q_eval.device)))
        
        actions, hidden_ = self.q_eval.forward(state.unsqueeze(0), self.hidden)
        # choose-action alters agent
        self.hidden = hidden_

        if np.random.random() > epsilon:
            # w.p. 1-epsilon, greedy
            # alternative: softmax
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


    # implement polyak soft update 
    # - unstable performance when directly copying target network
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            #self.q_next.load_state_dict(self.q_eval.state_dict())
            polyak_update(self.q_eval.parameters(), self.q_next.parameters(), self.tau)


    ### Polyak soft update

    

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cnt < self.batch_size:
            return None

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()

        actions_onehot = F.one_hot(actions, self.n_actions)
        actions_onehot = T.concat((T.zeros((self.batch_size, 1, self.n_actions)), actions_onehot[:,:-1,:]), dim=1)
        output = T.concat((actions_onehot, states), dim=2)

        q_pred = T.gather(self.q_eval.forward(output)[0], 2, actions.unsqueeze(2)).squeeze(2)
        actions_onehot = F.one_hot(actions, self.n_actions)
        actions_onehot = T.concat((T.zeros((self.batch_size, 1, self.n_actions)), actions_onehot), dim=1)
        temp = T.concat((actions_onehot, T.concat((states[:,0,].unsqueeze(1), states_), dim=1)), dim=2)
        q_next = self.q_next.forward(temp)[0][:,1:,:]
        q_eval = self.q_eval.forward(temp)[0][:,1:,:].detach()

        max_actions = T.argmax(q_eval, dim=2)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*T.gather(q_next, 2, max_actions.unsqueeze(2)).squeeze(2)

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss

 