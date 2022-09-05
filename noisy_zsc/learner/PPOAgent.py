import torch as T
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .reply_memory import RolloutBuffer

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
        
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
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

    def __init__(self, gamma lr, n_actions, input_dims, stdev, timesteps_per_batch=4800, episode_length=1600):
        self.actor = PPONetwork(lr, n_actions, input_dims)
        self.critic = PPONetwork(lr, 1, input_dims)
        self._init_hyperparameters()
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.stdev = stdev
        self.gamma = gamma
        self.timesteps_per_batch = timesteps_per_batch            # timesteps per batch
        self.max_timesteps_per_episode = episode_length      # timesteps per episode

        self.memory = RolloutBuffer


        self.cov_var = T.full(size=(self.n_actions,), fill_value=stdev)
        # Create the covariance matrix
        self.cov_mat = T.diag(self.cov_var)
    

    # compute reward-to-go
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


    def choose_action(self, obs, epsilon = None):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor.forward(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return the sampled action and the log prob of that action
        return action.detach().numpy(), log_prob.detach()
        

    def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.
			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)
			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""

		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs


    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones


    def learn(self, total_timesteps):
	    """
			Train the actor and critic networks. Here is where the main PPO algorithm resides.
			Parameters:
				total_timesteps - the total number of timesteps to train for
			Return:
				None
		"""
        t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.sample_memory()                 # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# Normalizing advantages to decrease variance, makes convergence much more stable and faster
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# update network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				ratios = T.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = T.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-T.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()


