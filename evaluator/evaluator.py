# evaluator for xplay

from typing import Optional, List, Callable
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters

from ..learner import DQNLearner, policygradLearner

from ..game import noisy_lever_game

def eval_xplay(env: noisy_lever_game,
    agent1: DQNLearner,
    agent2: DQNLearner,
    train1: bool,
    train2: bool):

    joint_obs = env.reset()
    agent1.reset()
    agent2.reset()
    



