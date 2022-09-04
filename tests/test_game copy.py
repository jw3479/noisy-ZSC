from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner.DDRQNAgent import DDRQNAgent, DDRQNetwork
import torch as T
import numpy as np


# Environment parameters
mean_payoffs = [2., 4.]
sigma = 0
sigma1 = 0
sigma2 = 0
episode_length = 3


# Initialize environment
env = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)
q_net = DDRQNetwork(0.01, 2, env.obs_dim(), 4)
agent = DDRQNAgent(q_net, 0.99, 0.4, 0.01, 2, env.obs_dim(), 16, 8, 3)

for _ in range(100000):
    # Reset environment
    obs, _ = env.reset()
    last_action = None
    agent.hidden = None
    for step in range(3):
        action2 = 0 if step < 2 else 1
        action1 = agent.choose_action(obs[0], last_action)

        reward, done = env.step(action1,action2)
        print(f"Reward: {reward:3.2f}, epsilon: {agent.epsilon}")
        
        obs_ = env.get_obs()

        agent.store_transition(obs[0], action1, reward, obs_[0], done)
        agent.learn()

        # print(f'obs: {obs}, reward: {reward}, done: {done}')
        
        last_action = action1
        obs = obs_