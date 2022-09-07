from re import L
import torch.nn as nn # layers
import torch.nn.functional as F #activation function
import torch.optim as optim #optimizers
import torch as T # basic package
import numpy as np



import os

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cnt = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cnt % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cnt += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones


class EpisodeBuffer():
    def __init__(self, max_size, input_shape, episode_length):
        self.mem_cnt = 0
        self.trajectory_cnt = 0
        self.mem_size = max_size
        self.episode_length = episode_length
        self.state_memory = np.zeros((self.mem_size, episode_length, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, episode_length, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.episode_length), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, self.episode_length), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, self.episode_length), dtype=np.bool8)

    def store_transition(self, state, action, reward, state_, done):
        mem_index = self.mem_cnt % self.mem_size
        traj_index = self.trajectory_cnt % self.episode_length
        self.state_memory[mem_index, traj_index] = state
        self.action_memory[mem_index, traj_index] = action
        self.reward_memory[mem_index, traj_index] = reward
        self.new_state_memory[mem_index, traj_index] = state_
        self.terminal_memory[mem_index, traj_index] = done
        self.trajectory_cnt += 1
        self.mem_cnt += 1 if self.trajectory_cnt % self.episode_length == 0 else 0

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones


class RolloutBuffer():
    # added: log_probs for Policy Gradient
    def __init__(self, max_size, input_shape, episode_length):
        self.mem_cnt = 0
        self.trajectory_cnt = 0
        self.mem_size = max_size
        self.episode_length = episode_length

        self.state_memory = np.zeros((self.mem_size, episode_length, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, episode_length, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.episode_length), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, self.episode_length), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, self.episode_length), dtype=np.bool8)
        self.log_probs_memory = np.zeros((self.mem_size, self.episode_length, input_shape), dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, log_probs):
        mem_index = self.mem_cnt % self.mem_size
        traj_index = self.trajectory_cnt % self.episode_length
        self.state_memory[mem_index, traj_index] = state
        self.action_memory[mem_index, traj_index] = action
        self.reward_memory[mem_index, traj_index] = reward
        self.new_state_memory[mem_index, traj_index] = state_
        self.terminal_memory[mem_index, traj_index] = done
        self.log_probs_memory[mem_index, traj_index] = log_probs
        self.trajectory_cnt += 1
        self.mem_cnt += 1 if self.trajectory_cnt % self.episode_length == 0 else 0

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        log_probs = self.log_probs_memory[batch]
        return states, actions, rewards, states_, dones, log_probs


    def rollout(self):
        return self.state_memory, self.action_memory, self.log_probs_memory, self.reward_memory, self.terminal_memory 


