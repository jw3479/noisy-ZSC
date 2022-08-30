from game import NoisyLeverGame

import torch
import torch.nn as nn
import random

def run():
    game = NoisyLeverGame(
        mean_payoffs=[1., 2., 3.],
        sigma=0.1,
        sigma1=0.5,
        sigma2=1,
        episode_length=3
    )

    # TODO: Setup agents here
    policy_net1 = nn.Sequential(
        nn.Linear(game.obs_dim(), 16),
        nn.ReLU(),
        nn.Linear(16, game.n_actions()),
        nn.Softmax(dim=0),
    )

    policy_net2 = nn.Sequential(
        nn.Linear(game.obs_dim(), 16),
        nn.ReLU(),
        nn.Linear(16, game.n_actions()),
        nn.Softmax(dim=0),
    )

    for epoch in range(1000):
        joint_obs, true_state = game.reset()
        # print(f'True world E*: {true_state}\n')

        obs1 = joint_obs[0]
        obs2 = joint_obs[1]
        # print(f'Obs1: {obs1}, Obs2: {obs2}\n')

        done = False
        while not done:
            # TODO: Make actions function of a learning policy
            # action1 = random.randrange(0, game.n_actions())
            # action2 = random.randrange(0, game.n_actions())
            action1 = torch.argmax(policy_net1(torch.tensor(obs1)))
            action2 = torch.argmax(policy_net2(torch.tensor(obs2)))

            reward, done = game.step(action1, action2)
            # print(f'(A1, A2): ({action1}, {action2}) -> Reward: {reward}, Done: {done}')

            joint_obs = game.get_obs()
            obs1 = joint_obs[0]
            obs2 = joint_obs[1]
            # print(f'Obs1: {obs1}, Obs2: {obs2}', end='\n' if done else '\n\n')

            # TODO: Give experience to learners
            # TODO: Train learners


#if __name__ == '__main__':
    #random.seed(42)
    #run()

