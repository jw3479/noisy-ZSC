import torch as T
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class PPONetwork(nn.Module):
    """
    network for PPOAgent
    """
    def __init__(self, lr, n_actions, input_dims, hidden_units = 16):
        super(PPONetwork).__init__()
        # first test for feed-forward nn
        n_hidden_units = hidden_units
        self.fc1 = nn.Linear(input_dims, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, hidden=None):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    """
    PPOAgent
    """

    def __init__(self, lr, n_actions, input_dims, stdev):
        self.actor = PPONetwork(lr, n_actions, input_dims)
        self.critic = PPONetwork(lr, 1, input_dims)
        self._init_hyperparameters()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.stdev = stdev

        self.cov_var = T.full(size=(self.n_actions,), fill_value=stdev)
        # Create the covariance matrix
        self.cov_mat = T.diag(self.cov_var)


    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode

    


    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = T.tensor(batch_rtgs, dtype=T.float)
        return batch_rtgs


    def choose_action(self, observation, epsilon = None):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()
        

    def learn(self, total_timesteps):
        t_so_far = 0 # timesteps simulated so far
        while t_so_far < total_timesteps:
            # TODO


