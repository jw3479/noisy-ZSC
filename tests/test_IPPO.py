from noisy_zsc.game import NoisyLeverGame
from noisy_zsc.learner.PPOAgent import PPOAgent
import torch as T
import numpy as np
import matplotlib.pyplot as plt


# Environment parameters
mean_payoffs = [5., 5., 5., 5.]
sigma = 1
sigma1 = 0
sigma2 = 0
episode_length = 1


# hyper-parameters
#lr = 0.005
lr = 0.0003
n_epochs = 10 # 3-10
clip = 0.2 # typical 0.1-0.3
N = 20

batch_size = 64


# Initialize environment
env = NoisyLeverGame(mean_payoffs, sigma, sigma1, sigma2, episode_length)
n_actions = len(env.mean_payoffs)
obs_dim = env.obs_dim()

agent = PPOAgent(n_actions=n_actions, batch_size=batch_size, 
                    alpha=lr, n_epochs=n_epochs, 
                    input_dims=obs_dim)

n_games = 100000


n_steps = 0
learn_iters = 0
avg_score = 0
score_history = []


def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig("test.png")


for epi in range(n_games):
    obs, _ = env.reset()
    done = False
    score = 0
    while not done:
        action2 = np.argmax(env.true_payoffs)
        #action2 = 1
        action, prob, val = agent.choose_action(obs[0])
        reward, done = env.step(action, action2)
        obs_ = env.get_obs()
        n_steps += 1
        score += reward
        agent.remember(obs[0], action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        obs = obs_
    
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    print('episode', epi, 'score %.1f' % score, 'avg score %.1f' % avg_score)
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history)