# Independent PPO

from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner.PPOAgent import PPOAgent
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import wandb
from copy import deepcopy
import random
from collections import deque

def run():
    wandb.init()
    config = wandb.config
    wandb.run.name = f"sigma={config.sigma}"
    mean_payoffs = [5., 5., 5., 5.]
    sigma = config.sigma
    sigma1 = 0
    sigma2 = 0
    episode_length = 1

    # hyper-parameters
    lr = config.learning_rate
    n_epochs = config.n_epochs # 3-10
    clip = config.clip # typical 0.1-0.3
    N = 20
    batch_size = 64


    # Initialize environment
    env = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)
    n_actions = len(env.mean_payoffs)
    obs_dim = env.obs_dim()
    agent1 = PPOAgent(n_actions=n_actions, batch_size=batch_size, 
                        alpha=lr, n_epochs=n_epochs, 
                        input_dims=obs_dim, policy_clip=clip)
    #agent2 = deepcopy(agent1)
    
    n_games = 100000

    n_steps = 0
    score_list = []
    avg_score = 0
    loss_list = []
    epi_list = []
    # action buffer to interpret agent policies
    action_buffer = deque(maxlen = 100)

    for epi in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0

        while not done:
            action2 = np.argmax(env.true_payoffs)
            action1, prob, val = agent1.choose_action(obs[0])
            
            # add actions to action_buffer 
            if action1 == action2:
                action_buffer.append(action1)
            else:
                action_buffer.append(-1)
            
            reward, done = env.step(action1, action2)

            obs_ = env.get_obs()
            n_steps += 1
            score += reward

            agent1.remember(obs[0], action1, prob, val, reward, done)
            #agent2.remember(obs[1], action2, prob, val, reward, done)

            if n_steps % N == 0:
                agent1.learn()
                #agent2.learn()

            obs = obs_
        
        score_list.append(score)
        avg_score = np.mean(score_list[-100:])
        
        stats = {
            "avg_score": avg_score,
            "reward": score / (env.episode_length * max(env.true_payoffs)),
            "prop_1": sum(1 for a in action_buffer if a == 0) / len(action_buffer),
            "prop_2": sum(1 for a in action_buffer if a == 1) / len(action_buffer),
            "prop_3": sum(1 for a in action_buffer if a == 2) / len(action_buffer),
            "prop_NA": sum(1 for a in action_buffer if a == -1) / len(action_buffer)}   
        wandb.log(stats)



if __name__ == '__main__':
    random.seed(42)
    run()
