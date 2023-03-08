import numpy as np

'''
Reward func calculates the reward for the agent
'''
def inverse_reward_func(array):
    if np.sum(array) == 1:
        return 1
    elif np.sum(array) == 0:
        return 0
    else:
        return 1/np.sum(array)

def span_reward_function(array):
    pass