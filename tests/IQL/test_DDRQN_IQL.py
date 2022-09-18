from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner.DDRQNAgent import DDRQNAgent, DDRQNetwork
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import wandb
import random
from copy import deepcopy
from collections import deque

def run():
    wandb.init()
    config = wandb.config
    wandb.run.name = f"epi_l={config.episode_length}-sigma={config.sigma}-tau={config.tau}"
    
    # Environment parameters
    #mean_payoffs=[100.,100.,100.]
    mean_payoffs=[100.,100.]
    sigma=config.sigma
    sigma1 = 0
    sigma2 = 0
    episode_length=config.episode_length


    # Initialize environment
    game = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)

    q_net = DDRQNetwork(lr=config.learning_rate,n_actions=len(game.mean_payoffs), 
                        input_dims=game.obs_dim(), hidden_units=4)

    agent1 = DDRQNAgent(q_eval = q_net, gamma = 0.99, epsilon = config.epsilon, 
                        n_actions=len(game.mean_payoffs), 
                        input_dims = game.obs_dim(), mem_size = config.mem_size, 
                        batch_size = config.batch_size, episode_length = game.episode_length, tau = config.tau, 
                        eps_min = config.eps_min, eps_dec = config.eps_dec)
    
    agent2 = deepcopy(agent1)

    epi_list = []
    loss_list = []
    reward_list = []

    # action buffer to interpret agent policies
    action_buffer = deque(maxlen = 100)

    for episode in range(100000):
        epi_reward = 0
        epi_list.append(episode)

        # Reset game
        obs, _ = game.reset()

        last_action1 = None
        last_action2 = None

        agent1.hidden = None
        agent2.hidden = None
        
        done = False 
        cnt = 0

        while not done:
            
            action1 = agent1.choose_action(obs[0], last_action1)
            action2 = agent2.choose_action(obs[1], last_action2)

            # add actions to action_buffer 
            if action1 == action2:
                action_buffer.append(action1)
            else:
                action_buffer.append(-1)

            reward, done = game.step(action1,action2)

            print(f"Reward: {reward:3.2f}, epsilon: {agent1.epsilon}")
            
            epi_reward += reward
            obs_ = game.get_obs()

            # store agent memory
            agent1.store_transition(obs[0], action1, reward, obs_[0], done)
            agent2.store_transition(obs[1], action2, reward, obs_[1], done)

            loss1 = agent1.learn()
            loss2 = agent2.learn()

            if loss1 is None:
                loss_list.append(0)
            else:
                loss_list.append(loss1.item())
            # print(f'obs: {obs}, reward: {reward}, done: {done}')
            
            last_action1 = action1
            last_action2 = action2
            obs = obs_

        reward_list.append(epi_reward)

        stats = {
            "reward": epi_reward / (game.episode_length * max(game.true_payoffs)),
            "prop_1": sum(1 for a in action_buffer if a == 0) / len(action_buffer),
            "prop_2": sum(1 for a in action_buffer if a == 1) / len(action_buffer),
            "prop_3": sum(1 for a in action_buffer if a == 2) / len(action_buffer),
            "prop_NA": sum(1 for a in action_buffer if a == -1) / len(action_buffer)}   

        wandb.log(stats)



#%%       
if __name__ == '__main__':
    random.seed(42)
    run()

# %%

