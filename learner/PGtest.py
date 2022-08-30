import gym
import matplotlib.pyplot as plt
import numpy as np
from policygradLearner import PGLearner

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100): (i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()


env = gym.make('LunarLander-v2')

n_games = 3000

agent = PGLearner(gamma = 0.99, lr = 0.0005, input_dims=[8], n_actions = 4)
scores = []
for i in range(n_games):
    done = False
    observation = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_rewards(reward)
        observation= observation_
    agent.learn()
    scores.append(score)

    avg_score = np.mean(scores[-100:])
    print('episode ', i, 'score %.2f' %score, 'avg score %.2f' % avg_score)
x = [i+1 for i in range(len(scores))]
plot_learning_curve(scores, x)



for i in range(n_games):
    obs = env.reset()
    score = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs_, reward, done, info = env.step(action)
        score += reward
        #env.render()

    print('episode ', i, 'score %.1f' % score)
