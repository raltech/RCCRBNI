import numpy as np

'''
Reward func calculates the reward for the agent
'''
def inverse_reward_func(array, dict_hat, dict_hat_count):
    if np.sum(array) == 1:
        return 1
    elif np.sum(array) == 0:
        return 0
    else:
        return 1/np.sum(array)
    
def more_cell_reward_func(array, dict_hat, dict_hat_count):
    return np.sum(array)

def scatter_reward_function(array, dict_hat, dict_hat_count):
    # calculate the average variance associated with each action
    avg_var = np.mean(np.var(dict, axis=1))

    # calculate the eigenvalues of scatter matrix
    eigvals = np.linalg.eigvals(np.transpose(dict) @ dict)

    return np.sum(eigvals[eigvals < avg_var]).real

def diversity_reward_function(array, dict_hat, dict_hat_count):
    reward = 0.0
    nonzero_idx = np.nonzero(array)[0]
    if len(nonzero_idx) == 0:
        reward = 0
    else:
        for idx in nonzero_idx:
            tmp = np.nonzero(dict_hat[:, idx])[0]
            reward += 1/np.sum(dict_hat[tmp, idx] / dict_hat_count[tmp])
    return reward
    