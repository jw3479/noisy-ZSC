from .lever_game import LeverGame
from typing import List, Tuple
from random import normalvariate
import numpy as np


class NoisyLeverGameWithOtherPlay:
    
    def __init__(self, mean_payoffs, sigma, sigma1, sigma2, episode_length):
        self.mean_payoffs = mean_payoffs
        self.episode_length = episode_length

        self.sigma = sigma
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.true_lever_game = None
        self.true_payoffs = None
        self.payoffs1 = None
        self.payoffs2 = None

        self.reset()

    def reset(self) -> Tuple:
        # E* (true_payoffs) is drawn from gaussian distribution around mean_payoffs
        self.true_payoffs = np.array([max(0.01, normalvariate(payoff, self.sigma)) for payoff in self.mean_payoffs])

        perm1 = np.random.permutation(self.true_payoffs.shape[0])
        perm2 = np.random.permutation(self.true_payoffs.shape[0])
        self.true_payoffs1 = self.true_payoffs[perm1].tolist()
        self.true_payoffs2 = self.true_payoffs[perm2].tolist()
        self.true_payoffs  = self.true_payoffs.tolist()

        self.perm1, self.perm2 = perm1, perm2

        # E_A (payoff1) and E_B (payoff2) are noisy versions of true_lever_game (adding noise to reward)
        self.payoffs1 = tuple([normalvariate(payoff, self.sigma1) for payoff in self.true_payoffs1])
        self.payoffs2 = tuple([normalvariate(payoff, self.sigma2) for payoff in self.true_payoffs2])

        self.true_lever_game = LeverGame(payoffs=self.true_payoffs, episode_length=self.episode_length)

        self.true_lever_game.reset()
        return self.get_obs(), self.true_lever_game.payoffs

    def step(self, action1: int, action2: int) -> Tuple[float, bool]:
        action1 = self.perm1[action1]
        action2 = self.perm2[action2]
        return self.true_lever_game.step(action1, action2)

    def get_obs(self) -> List[Tuple]:
        true_obs1, true_obs2 = self.true_lever_game.get_obs()
        #return [self.payoffs1 + (true_obs1,self.sigma, self.sigma1, self.sigma2,self.true_lever_game.episode_step,),
        #        self.payoffs2 + (true_obs2,self.sigma, self.sigma1, self.sigma2,self.true_lever_game.episode_step,)]
        
        #obs = [self.payoffs1 + (true_obs1,self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0]),
        #        self.payoffs2 + (true_obs2,self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0])]
        
        # add observation of mean reward
        #obs = [self.payoffs1 + (self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0]),
        #        self.payoffs2 + (self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0])]
        obs = [self.payoffs1 + (self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0]),
               self.payoffs2 + (self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0])]
        
        return obs

    def is_terminal(self) -> bool:
        return self.true_lever_game.is_terminal()

    def n_actions(self):
        return self.true_lever_game.n_actions()

    def obs_dim(self):
        return len(self.get_obs()[0])


if __name__ == '__main__':
    num_levers = 4
    g = NoisyLeverGameWithOtherPlay([5]*num_levers, 1.0, 0.1, 0.1, 1)
    g.reset()
    random_policy = lambda : np.random.randint(num_levers)
    for i in range(100):
        x = g.step(random_policy(), random_policy())
        print(x)
        g.reset()

