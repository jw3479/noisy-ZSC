#%%
from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner.PPOAgent import PPOAgent, ActorNetwork, CriticNetwork
import torch as T
import numpy as np
import matplotlib.pyplot as plt

# Environment parameters
mean_payoffs = [500., 500., 500.]
sigma = 100
sigma1 = 0
sigma2 = 0
episode_length = 1


# hyper-parameters
#lr = 0.005
lr = 0.005

# Initialize environment
env = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)
n_actions = len(env.mean_payoffs)

# actor: input observation, output probability of taking each action
actor_net = ActorNetwork(lr=lr, output_dims=n_actions, input_dims=env.obs_dim())

# critic: input observation, output current value function estimate
critic_net = CriticNetwork(lr=lr, output_dims=1, input_dims=env.obs_dim())

agent = PPOAgent(actor = actor_net, critic=critic_net, gamma = 0.99, n_actions = n_actions, 
            input_dims = env.obs_dim(), mem_size = 1, episode_length= episode_length)

for training_step in range(100000):
    # fill up rollout buffer 
    for episode in range(agent.mem_size):
        obs, _ = env.reset()
        for step in range(episode_length):
            action2 = np.argmax(env.true_payoffs)
            action1, log_prob = agent.choose_action(obs[0])
            reward, done = env.step(action1,action2)
            obs_ = env.get_obs()
            agent.store_transition(obs[0], action1, reward, obs_[0], done, log_prob)
            obs = obs_
    
    c_loss, a_loss = agent.learn()

    obs, _ = env.reset()
    # evaluate
    cum_rew = 0
    for step in range(episode_length):
        action2 = np.argmax(env.true_payoffs)
        #print(f'actor: {agent.actor(T.tensor(obs[0]))}')
        action1, log_prob = agent.choose_action(obs[0])
        reward, done = env.step(action1,action2)
        cum_rew += reward
        obs_ = env.get_obs()
        #agent.store_transition(obs[0], action1, reward, obs_[0], done, log_prob)
        obs = obs_

    #print(f'{training_step} -- return: {cum_rew:3.0f} -- loss: ({c_loss:.3f},{a_loss:.3f})')
    