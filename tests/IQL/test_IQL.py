# testing vanilla independent Q-learning (IQL) with double DQN learning
# i.e. each agent treats partner as part of the environment
# without reasoning about joint actions

#%%
import torch as T
import numpy as np
from copy import deepcopy
import random

from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner import DDQNAgent
import matplotlib.pyplot as plt

def convert_dec(myList):
    return list(np.around(np.array(myList),2))

def run():
    # lever game with noisy true-lever game (gaussianm noise)
    game = NoisyLeverGame(
        mean_payoffs=[2., 2., 2.],
        sigma=0.5,
        sigma1=0,
        sigma2=0,
        episode_length=5
    )

    agent1 = DDQNAgent(gamma=1.0, epsilon=0.5, lr=0.001, n_actions=len(game.mean_payoffs),
                       input_dims=game.obs_dim(), mem_size=4048,
                       batch_size=256, eps_min=0.01,
                       eps_dec=5e-5, replace=250,
                       algo=None, env_name="noisy_lever", chkpt_dir = 'tmp/dqn')

    agent2 = deepcopy(agent1)
    epi_reward_list = []
    epi_list = []
    epi_loss_list = []

    for episode in range(100000):
        epi_reward = 0
        #print(f'episode: {episode}')
        joint_obs, true_state = game.reset()
        #print(f'True world E*: {convert_dec(true_state)}\n')

        obs1 = joint_obs[0]
        obs2 = joint_obs[1]
        #print(f'Obs1: {convert_dec(obs1)}, Obs2: {convert_dec(obs2)}\n')
        if (episode+1) % 500 == 0:
            print(f'episode: {episode}')
        done = False
        cnt = 0
        while not done:
            obs1 = joint_obs[0]
            obs2 = joint_obs[1]

            action1 = agent1.choose_action(obs1)
            action2 = agent2.choose_action(obs2)
            
            reward, done = game.step(action1, action2)
            epi_reward += reward
            if done:
                epi_reward_list.append(epi_reward / (game.episode_length * max(game.true_payoffs)))
                epi_list.append(episode)
            #print(f'(A1, A2): ({action1}, {action2}) -> Reward: {np.around(reward,2)}, Done: {done}')

            joint_obs_ = game.get_obs()
            obs1_ = joint_obs_[0]
            obs2_ = joint_obs_[1]
            #print(f'Obs1: {convert_dec(obs1)}, Obs2: {convert_dec(obs2)}', end='\n' if done else '\n\n')

            # Give experience to learners
            agent1.store_transition(obs1, action1, reward, obs1_, done)
            agent2.store_transition(obs2, action2, reward, obs2_, done)

            # Train learners
            loss1 = agent1.learn_old()
            loss2 = agent2.learn_old()
            joint_obs = joint_obs_

            if loss1 is not None:
                epi_loss_list.append(loss1.item())
            else:
                epi_loss_list.append(2.0)

            #if (episode+1) % 500 == 0:
            #    print(f'obs1: {obs1}, q-value: {agent1.q_eval.forward(T.tensor(obs1))}')

            cnt += 1

        if episode % 50 == 0:
            #plt.plot(epi_loss_list)
            plt.scatter(epi_list, epi_reward_list)
            #plt.semilogy()
            plt.show(block=False)
            #plt.show()
            plt.pause(0.1)
            plt.close()

    return epi_reward_list
#%%

random.seed(42)
reward_list = run()

# example where DRQN > DQN
# fist step with Dec-POMDP

# Q value don't converge? add time step as obs