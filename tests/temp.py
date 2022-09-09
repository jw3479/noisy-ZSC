import torch as T
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import torch.optim as optim
import time


def evaluate(self, batch_obs, batch_actions):
        V = []
        log_probs = []

        for traj_index in range(self.mem_size):
            traj_obs = batch_obs[traj_index]
            traj_actions = batch_actions[traj_index]
            traj_V = self.critic(T.tensor(traj_obs)).squeeze()
            action_probs = F.softmax(self.actor.forward(T.tensor(traj_obs)),dim=0)
            dist=T.distributions.categorical.Categorical(probs=action_probs)
            traj_log_probs = dist.log_prob(T.tensor(traj_actions))
            V.append(traj_V)
            log_probs.append(traj_log_probs)

        return V, log_probs


a = T.randn([2, 3]).unsqueeze(0)
b = T.randn([2, 3]).unsqueeze(0)
print(T.concat([a,b], dim = 0).shape)