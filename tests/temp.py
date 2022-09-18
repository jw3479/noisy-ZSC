from ast import arg
import numpy as np


def argmax_of_2_score(payoffs1, payoffs2, true_payoff):
    action1 = np.argmax(payoffs1[1:])
    action2 = np.argmax(payoffs2[1:])
    if action1 == action2:
        return true_payoff[action1+1]
    else:
        return 0

true_payoff = [7,4,5,6]
payoffs1 = true_payoff
payoffs2 = true_payoff

print(argmax_of_2_score(payoffs1, payoffs2, true_payoff))