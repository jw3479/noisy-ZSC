from typing import List, Tuple


# LeverGame setup:
# - two players; payoff of lever according to payoffs; with episode_length
# - observation (current): at time t, each player observes the last action of their opponent at t-1
# - alternative observation (harder), each player observes a boolean indicating
# whether the last action was a match

class LeverGame:

    def __init__(self, payoffs: List, episode_length: int):
        self.payoffs = payoffs
        self.episode_step = 0
        self.episode_length = episode_length
        self.last_action1 = None
        self.last_action2 = None

    def reset(self) -> List:
        self.episode_step = 0
        self.last_action1 = None 
        self.last_action2 = None
        return self.get_obs()

    def step(self, action1: int, action2: int) -> Tuple[float, bool]:
        # TODO: Check if action is valid
        self.last_action1 = action1
        self.last_action2 = action2
        self.episode_step += 1
        reward = self.payoffs[action1] if action1 == action2 else 0
        return (reward, self.is_terminal())

    def get_obs(self) -> List[int]:
        if self.episode_step == 0: 
            # at time 0 no action yet and no observation (0 indicates null)
            return [0, 0]
        else:
            # observation (harder): indicator of whether the last action was a match
            #did_match = self.last_action1 == self.last_action2
            #return [int(did_match), int(did_match)]

            # observation (current): last action of partner
            #return [self.last_action2, self.last_action1]

            # single episode, no observation
            return [0,0]

    def is_terminal(self) -> bool:
        return self.episode_step >= self.episode_length

    def n_actions(self):
        return len(self.payoffs)

    def obs_dim(self):
        return 1


