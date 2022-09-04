#%%
from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner.DDRQNAgent import DDRQNAgent, DDRQNetwork
import torch as T
import numpy as np
import matplotlib.pyplot as plt

# Environment parameters
mean_payoffs = [2., 4.]
sigma = 0
sigma1 = 0
sigma2 = 0
episode_length = 3


# Initialize environment
env = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)
q_net = DDRQNetwork(0.01, 2, env.obs_dim(), 4)
agent = DDRQNAgent(q_eval = q_net, gamma = 0.99, epsilon = 0.4, lr = 0.01, 
                    n_actions = 2, input_dims = env.obs_dim(), mem_size = 16, 
                    batch_size = 8, episode_length = 3, tau = 0.1, eps_min = 0.005, eps_dec = 1e-5)

epi_list = []
loss_list = []
reward_list = []
for episode in range(100000):
    epi_list.append(episode)
    # Reset environment
    obs, _ = env.reset()
    last_action = None
    agent.hidden = None
    total_reward = 0
    for step in range(3):
        action2 = 0 if step < 2 else 1
        action1 = agent.choose_action(obs[0], last_action)

        reward, done = env.step(action1,action2)

        print(f"Reward: {reward:3.2f}, epsilon: {agent.epsilon}")
        total_reward += reward
        obs_ = env.get_obs()

        agent.store_transition(obs[0], action1, reward, obs_[0], done)
        loss1 = agent.learn()
        if loss1 is None:
            loss_list.append(0)
        else:
            loss_list.append(loss1.item())
        
        
        # print(f'obs: {obs}, reward: {reward}, done: {done}')
        
        last_action = action1
        obs = obs_
    reward_list.append(total_reward)


    if episode % 50 == 0:
        #plt.plot(epi_loss_list)
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.scatter(epi_list, reward_list)
        ax2.plot(loss_list)
        #plt.semilogy()
        #plt.show(block=False)
        
        plt.savefig("hi.png")
        plt.close()
"""    
fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter(epi_list, reward_list)
ax2.plot(loss_list)
#plt.semilogy()
#plt.show(block=False)
plt.show()
#plt.pause(0.1)
plt.close()

"""    
    



# %%

