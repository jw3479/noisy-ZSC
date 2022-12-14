from .lever_game import LeverGame
from typing import List, Tuple
from random import normalvariate


class NoisyBailLeverGame:
    
    def __init__(self, mean_payoffs, bail_payoff, sigma, sigma1, sigma2, episode_length):
        self.mean_payoffs = mean_payoffs
        self.bail_payoff = bail_payoff
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
        # allowing negative reward for bad levers
        self.true_payoffs = [normalvariate(payoff, self.sigma) for payoff in self.mean_payoffs]

        # E_A (payoff1) and E_B (payoff2) are noisy versions of true_lever_game (adding noise to reward)
        self.payoffs1 = tuple([normalvariate(payoff, self.sigma1) for payoff in self.true_payoffs]+[self.bail_payoff])
        self.payoffs2 = tuple([normalvariate(payoff, self.sigma2) for payoff in self.true_payoffs]+[self.bail_payoff])
        
        self.true_payoffs = self.true_payoffs + [self.bail_payoff]

        self.true_lever_game = LeverGame(payoffs=self.true_payoffs, episode_length=self.episode_length)

        self.true_lever_game.reset()
        return self.get_obs(), self.true_lever_game.payoffs

    def step(self, action1: int, action2: int) -> Tuple[float, bool]:
        return self.true_lever_game.step(action1, action2)


    def get_obs(self) -> List[Tuple]:
        true_obs1, true_obs2 = self.true_lever_game.get_obs()
        #return [self.payoffs1 + (true_obs1,self.sigma, self.sigma1, self.sigma2,self.true_lever_game.episode_step,),
        #        self.payoffs2 + (true_obs2,self.sigma, self.sigma1, self.sigma2,self.true_lever_game.episode_step,)]
        #return [self.payoffs1 + (true_obs1,self.sigma, self.sigma1, self.sigma2,),
        #        self.payoffs2 + (true_obs2,self.sigma, self.sigma1, self.sigma2,)]
        return [self.payoffs1 + (self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0],),
                self.payoffs2 + (self.sigma, self.sigma1, self.sigma2,self.mean_payoffs[0],)]

    def is_terminal(self) -> bool:
        return self.true_lever_game.is_terminal()

    def n_actions(self):
        return self.true_lever_game.n_actions()

    def obs_dim(self):
        return len(self.get_obs()[0])


