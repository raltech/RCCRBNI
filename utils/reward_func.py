import numpy as np

'''
Reward func calculates the reward for the agent
'''
def inverse_reward_func(array, dict):
    if np.sum(array) == 1:
        return 1
    elif np.sum(array) == 0:
        return 0
    else:
        return 1/np.sum(array)

def scatter_reward_function(array, dict):
    # calculate the average variance associated with each action
    avg_var = np.mean(np.var(dict, axis=1))

    # calculate the eigenvalues of scatter matrix
    eigvals = np.linalg.eigvals(np.transpose(dict) @ dict)

    return np.sum(eigvals[eigvals < avg_var]).real

def diversity_reward_function(array, dict):
    