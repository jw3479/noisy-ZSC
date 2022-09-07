import torch as T
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .reply_memory import RolloutBuffer
import torch.optim as optim
import time

class PPONetwork(nn.Module):
    """
    network for PPOAgent
    """
    def __init__(self, lr, n_actions, input_dims, hidden_units = 8):
        super(PPONetwork, self).__init__()
        # first test for feed-forward nn
        n_hidden_units = hidden_units

        self.fc1 = nn.Linear(input_dims, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, hidden=None):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#self.actor = PPONetwork(lr, n_actions, input_dims)
#self.critic = PPONetwork(lr, 1, input_dims)
        
class PPOAgent():
    """
    PPOAgent
    """
    def __init__(self, actor: PPONetwork, critic: PPONetwork, gamma, n_actions, input_dims, mem_size,
                 episode_length, n_updates_per_iteration=5, clip = 0.2, stdev=0.5):
        self.actor = actor
        self.critic = critic            
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.stdev = stdev
        self.gamma = gamma
        self.mem_size = mem_size        
        self.episode_length = episode_length   
        self.n_updates_per_iteration = n_updates_per_iteration
        self.clip = clip

        self.memory = RolloutBuffer(mem_size, input_dims, episode_length)


        self.cov_var = T.full(size=(self.n_actions,), fill_value=stdev)
        # Create the covariance matrix
        self.cov_mat = T.diag(self.cov_var)
        # This logger will help us with printing out summaries of each iteration
        self.logger = {
			'delta_t': time.time_ns(),
			'episode': 0,          
			'batch_rews': [],      
			'actor_losses': [],    
		}
    

    # compute reward-to-go: matrix
    def compute_rtgs(self, batch_rews):

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = T.tensor(batch_rtgs, dtype=T.float)

        return batch_rtgs

    
    def choose_action(self, obs):
        # Query the actor network for a mean action
        mean = self.actor.forward(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
    
        # Return the sampled action and the log prob of that action
        return action.detach().numpy(), log_prob.detach()
    

    def evaluate(self, batch_obs, batch_actions):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)
        return V, log_probs

    def store_transition(self, state, action, reward, state_, done):
        # compute log prob based on actor network
        log_probs = None
        self.memory.store_transition(state, action, reward, state_, done, log_probs)

    def learn(self):
        epi_cnt = 0
        while epi_cnt < self.mem_size:
            batch_obs, batch_acts, batch_log_probs, batch_rewards, dones = self.memory.rollout()
            batch_rtgs = self.compute_rtgs(batch_rewards)

            self.logger['batch_rews'] = batch_rewards

            epi_cnt += 1
            self.logger['episode'] = epi_cnt

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()      
            # normalize A_k
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batcH_obs, batch_acts)
                ratios = T.exp(curr_log_probs - batch_log_probs)
                # PPO surrogate loss
                ratios = T.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = T.clamp(ratios, 1-self.clip, 1+self.clip) * A_k
                
                actor_loss = (-T.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()



    def _log_summary(self):
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))
        
        episode = self.logger['episode']
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {episode}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data