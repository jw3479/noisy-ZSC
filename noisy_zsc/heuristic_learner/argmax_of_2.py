# heuristic SP policy: agents both pick the max lever of two of the three random levers 
# and pull the higher lever of the two
# or bail if the max of the 2 is lower than bail


import numpy as np

class Argmaxof2Agent:
    def __init__(self):
        pass

    def choose_action(self, obs, n_levers):
        obs_payoffs = obs[0:n_levers]
        # ignore the first random lever
        action = np.argmax(obs_payoffs[1:n_levers])
        return action
