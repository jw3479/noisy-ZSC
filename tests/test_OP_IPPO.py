# Independent PPO
# both agents learn with PPO 
# note that the performance is very sensitive to hyperparameters and scale of the rewards
# fix mean_payoffs = [5., 5., 5.], and sigma in the ballpark of 1-3
# wandb logs agent strategies in the last 100 steps and how many hits at 
# max lever in the last 100 steps

from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.game import NoisyBailLeverGame
from noisy_zsc.game import NoisyLeverGameWithOtherPlay
from noisy_zsc.learner.PPOAgent import PPOAgent
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import wandb
from copy import deepcopy
import random
from collections import deque, namedtuple
import pandas as pd
import csv

#Config = namedtuple('Config', ['sigma', 'sigma1', 'sigma2', 'learning_rate', 'n_epochs', 'clip', 'ent_weight'])
#config = Config(0, 0.1, 0.1, 0.003, 2, 0.1, 0.01)

def run():
    wandb.init()
    config = wandb.config
    mean_payoffs = [5., 5., 5.]
    #bail_payoff = config.bail_payoff
    sigma = config.sigma
    sigma1 = config.sigma1
    sigma2 = config.sigma2
    episode_length = 1
    wandb.run.name = f"SBVF-sigma={config.sigma}-sigma1={config.sigma1}-sigma2={config.sigma2}"

    # hyper-parameters
    lr = config.learning_rate
    n_epochs = config.n_epochs # 3-10
    clip = config.clip # typical 0.1-0.3
    N = 20
    batch_size = 64

    # Initialize environment
#    env = NoisyBailLeverGame(mean_payoffs, bail_payoff, sigma, sigma1, sigma2, episode_length)
    env = NoisyLeverGameWithOtherPlay(mean_payoffs,sigma, sigma1, sigma2, episode_length)
    n_actions = len(env.true_payoffs)
    obs_dim = env.obs_dim()
    agent1 = PPOAgent(n_actions=n_actions, batch_size=batch_size, 
                        alpha=lr, n_epochs=n_epochs, 
                        input_dims=obs_dim, policy_clip=clip)
    agent2 = deepcopy(agent1)

    n_games = 1000000

    n_steps = 0

    score_list = []
    avg_score = 0
    loss_list = []
    epi_list = []

    # track the maximum value in true game, E1 and E2
    true_max = []
    E1_max = []
    E2_max = []

    # track actual lever (value) selected 
    reward_list = []
    E1_val = []
    E2_val = []

    # action buffer to interpret agent policies
    action_buffer = deque(maxlen = 100)
    max_lever_buffer = deque(maxlen = 100)
    bail_buffer = deque(maxlen = 100)

    for epi in range(n_games):
        obs, true_payoffs = env.reset()
        state = tuple(true_payoffs) + obs[1][len(true_payoffs):]

        #seq = sorted(env.payoffs1)
        #payoffs1_seq = [seq.index(p) for p in seq] 

        done = False
        score = 0
        max_lever = -1
        bail = -1
        epi_list.append(epi)

        while not done:

            action1, prob, val = agent1.choose_action(obs[0], state)
            action2, prob, val = agent2.choose_action(obs[1], state)

            true_max.append(np.max(env.true_payoffs))
            E1_max.append(np.max(env.payoffs1))
            E2_max.append(np.max(env.payoffs2))

            E1_val.append(env.payoffs1[action1])
            E2_val.append(env.payoffs2[action2])

            # add actions to action_buffer 
            if action1 == action2:
                action_buffer.append(action1)
                max_lever = 1 if action1 == np.argmax(env.true_payoffs) else 0 
                bail = 1 if action1 == len(env.true_payoffs) - 1 else 0
            else:
                action_buffer.append(-1)
                max_lever = 0

            max_lever_buffer.append(max_lever)
            bail_buffer.append(bail)

            reward, done = env.step(action1, action2)

            reward_list.append(reward)

            obs_ = env.get_obs()
            n_steps += 1
            score += reward

            agent1.remember(obs[0], action1, prob, val, reward, done)
            agent2.remember(obs[1], action2, prob, val, reward, done)

            if n_steps % N == 0:
                agent1.learn(ent_weight=config.ent_weight)
                agent2.learn(ent_weight=config.ent_weight)
            obs = obs_

        score_list.append(score)
        avg_score = np.mean(score_list[-100:])
        stats = {
            "avg_score": avg_score,
            "reward": score / (env.episode_length * max(env.true_payoffs)),
            "prop_1": sum(1 for a in action_buffer if a == 0) / len(action_buffer),
            "prop_2": sum(1 for a in action_buffer if a == 1) / len(action_buffer),
            "prop_3": sum(1 for a in action_buffer if a == 2) / len(action_buffer),
            "prop_NA": sum(1 for a in action_buffer if a == -1) / len(action_buffer),
            # number of max lever pulled in last 100 steps
            "max_lever": sum(1 for i in max_lever_buffer if i == 1) / len(max_lever_buffer), 
            "bail": sum(1 for i in bail_buffer if i == 1) / len(bail_buffer)} 

        wandb.log(stats)

"""
    df = pd.DataFrame({
        'true_max': true_max,
        'E1_max': E1_max,
        'E1_val': E1_val,
        'E2_max': E2_max,
        'E2_val': E2_val,
        'reward_list': reward_list})

"""


if __name__ == '__main__':
   #random.seed(42)
    run()
