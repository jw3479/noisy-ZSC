# heuristic (ZSC): both agents pull the max lever they observe
# ArgmaxAgent: always pull max observed lever

import numpy as np

class ArgmaxAgent:
    def __init__(self):
        pass

    def choose_action(self, obs, n_levers):
        obs_payoffs = obs[0:n_levers]
        action = np.argmax(obs_payoffs)
        return action
