from multiprocessing import Value
import torch as T
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .replay_memory import RolloutBuffer
import torch.optim as optim
import time

class ActorNetwork(nn.Module):
    """
    network for PPOAgent
    """
    def __init__(self, lr, output_dims, input_dims, hidden_units = 16):
        super(ActorNetwork, self).__init__()
        # first test for feed-forward nn
        n_hidden_units = hidden_units

        self.lr = lr

        self.fc1 = nn.Linear(input_dims, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, output_dims)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim = -1)


class CriticNetwork(nn.Module):
    """
    network for PPOAgent
    """
    def __init__(self, lr, output_dims, input_dims, hidden_units = 16):
        super(CriticNetwork, self).__init__()
        # first test for feed-forward nn
        n_hidden_units = hidden_units

        self.lr = lr

        self.fc1 = nn.Linear(input_dims, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, output_dims)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        


#self.actor = PPONetwork(lr, n_actions, input_dims)
#self.critic = PPONetwork(lr, 1, input_dims)
        
class PPOAgent():
    """
    PPOAgent
    """
    def __init__(self, actor: ActorNetwork, critic: CriticNetwork, 
                    gamma, n_actions, input_dims, mem_size,
                 episode_length, n_epochs=1, clip = 0.2):
        self.actor = actor
        self.critic = critic            
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size   
        self.episode_length = episode_length   
        self.n_epochs = n_epochs
        self.clip = clip
        # initialize RolloutBuffer
        self.memory = RolloutBuffer(mem_size, input_dims, episode_length)
    
    # compute reward-to-go: matrix
    # INPUT: batch_rew: size (mem_size, episode_length)
    # OUTPUT: batch_rtgs: size (mem_size, episode_length)
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = T.tensor(batch_rtgs, dtype=T.float)
        return batch_rtgs
    
    """
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for traj_index in range(self.mem_size):
            traj_rews = batch_rews[traj_index]
            traj_rtgs = []
            discounted_reward = 0.0
            for rew in reversed(traj_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                traj_rtgs.insert(0, discounted_reward)
            batch_rtgs.append(traj_rtgs)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = T.tensor(batch_rtgs, dtype=T.float)
        return batch_rtgs
    """
    
    
    # INPUT: single observation 
    # OUTPUT: 
    # action: single action by sampling stoc policy of actor, 
    # log_prob: log-prob according to actor policy 
    def choose_action(self, obs):
        # Query the actor network for a mean action
        action_probs = self.actor.forward(T.tensor(obs))
        
        dist=T.distributions.categorical.Categorical(probs=action_probs)
        
        action = dist.sample()
        
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        return action.detach().numpy(), log_prob.detach()
    

    # INPUT: 
    # batch observation - dim (mem_size, episode_length, obs_dim) 
    # batch action - dim (mem_size, episode_length)
    # OUTPUT:
    # 1) V: critic estimation of value
    # 2) log_probs: log prob of choosing action according to actor

    def evaluate(self, batch_obs, batch_actions):
        V_list = []
        log_probs_list = []

        for traj_index in range(self.mem_size):
            traj_obs = batch_obs[traj_index]
            traj_actions = batch_actions[traj_index]
            traj_V = self.critic(T.tensor(traj_obs)).squeeze()
            action_probs = self.actor.forward(T.tensor(traj_obs))
            dist=T.distributions.categorical.Categorical(probs=action_probs)
            traj_log_probs = dist.log_prob(T.tensor(traj_actions))
            V_list.append(traj_V.unsqueeze(0))
            log_probs_list.append(traj_log_probs.unsqueeze(0))
        V = T.concat(V_list)
        log_probs = T.concat(log_probs_list)
        return V, log_probs
        

    def store_transition(self, state, action, reward, state_, done, log_prob):
        # compute log prob based on actor network
        self.memory.store_transition(state, action, reward, state_, done, log_prob)

    def learn(self):
        # only call after memory
        # each call to learn() processes one batch of rollouts
        
        batch_obs, batch_acts, batch_log_probs, batch_rewards, dones = self.memory.rollout()
        # compute rewards-to-go for batch
    
        batch_rtgs = self.compute_rtgs(batch_rewards)

        V, _ = self.evaluate(batch_obs, batch_acts)

        A_k = batch_rtgs - V.detach()      
        # normalize A_k
        #A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(self.n_epochs):
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # PPO surrogate loss
            ratios = T.exp(curr_log_probs - T.tensor(batch_log_probs))
            surr1 = ratios * A_k
            surr2 = T.clamp(ratios, 1-self.clip, 1+self.clip) * A_k
            
            actor_loss = (-T.min(surr1, surr2)).mean()
            critic_loss = self.critic.loss(V, batch_rtgs)
    

            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

        #print(f'Loss_diff: {critic_loss.item()}, {actor_loss.item()}')
        return critic_loss.item(), actor_loss.item()
            

