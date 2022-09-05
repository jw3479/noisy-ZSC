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


class rolloutBuffer():
    # added: log_probs for Policy Gradient
    def __init__(self, max_size, input_shape, episode_length, log_probs):
        self.mem_cnt = 0
        self.trajectory_cnt = 0
        self.mem_size = max_size
        self.episode_length = episode_length
        self.log_probs = log_probs

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



def rollout(self):
        """
        DIMENSIONS:
            observations: (#timesteps per batch, obs_dim)
            actions: (#timesteps per batch, dimension of action)
            log probabilities: (#timesteps per batch)
            rewards: (#episodes, #timesteps per episode)
            reward-to-go’s: (#timesteps per batch)
            batch lengths: (#episodes)
        """

        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # t: timesteps run so far, up until self.timesteps_per_batch
        t = 0

        while t < self.timesteps_per_batch:

            # rewards for this episode
            epi_reward = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                # collect obs
                batch_obs.append(obs)
                
                action, log_prob = self.choose_action(obs)
                reward, done = self.env.step()
