# Noisy-ZSC
Codebase for project "Noisy Zero-Shot Coordination", supervised by Jakob Foerster at the University of Oxford.

## Overview

### Environment
- noisy_zsc/game/
    - lever_game.py: implements a standard 2-player lever game, where agent receive correponding reward iff pulling the same lever and 0 reward otherwise.  
    - noisy_bail_lever_game.py: implements the noisy lever game, a test bed for the noisy ZSC problem. The game consists of multiple levers with noisy reward, as well as bail lever with fixed reward. 
    - Each episode of the noisy lever game correpsonds to a meta episode of the noisy zsc problem: 
        - A ground true lever game is drawn from a distribution around mean_play
        - Both agents observe a noisy version fo the ground truth lever game at the beginning of the episode
        - The goal is for the two agents to pull the same lever which yield maximum reward in the true lever game

### Learners
We implement PPO, DQN, double DRQN for testing in the noisy lever game and eventually adopted independent PPO. 
- noisy_zsc/learner/
    - PPOAgent.py: PPO agent with entropy regularization.
    - DQNAgent.py: Double DQN agent augmented with experience replay, target network and polyak soft update to stabilize training.
    - DDRQNAgent.py: Recurrent Double DQN agent using LSTM layer for partial observability, augmented with experience replay, target network and polyak soft update to stabilize training.
    - replay_memory.py: experience replay buffer, used by the double DRQN agent. 

### Tests
- tests/
    - test_IPPO.py: independent PPO (IPPO) to test self-play on the noisy lever game. 
    - test_IQL.py: independent Q-learning (IQL) to test self-play on the noisy lever game with single episode (failed: stuck in local minimum e.g. stick with pulling first lever)
    - test_DDRQN_IQL.py: independent DDRQN to test self-play on the noisy lever game with single episode (failed: stuck in local minimum)
    - DDRQN_fixedPattern.py: sanity check on DDRQNAgent on a partner playing fixed pattern e.g. 1,2,3,1,2,3,... which requires observation history memory to solve perfectly. 

### Evaluators
- test/xplay/:
    - eval_IPPO.ipynb implements the cross-play matrix for independently trained IPPO agents using different seeds. 

### Baselines (heuristic learners)
- tests/heuristic_learner_values.ipynb:
    - implements the expected returns of three heuristic policies on the noisy lever game: including the bail policy, the stubborn policy and the argmax policy. 
    - Computes a simulated heatmap for visualizing the return of the argmax policy.
