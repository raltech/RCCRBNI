import numpy as np

'''
Reward func calculates the reward for the agent
'''
def max_cosine_sim_reward_func(array):
    dim = array.shape[0]
    # create a one hot vector for each dimension
    one_hot = np.eye(dim)
    # calculate the cosine similarity between the one hot vector and the array
    cos_sim = np.dot(one_hot, array)/(np.linalg.norm(one_hot, axis=1)*np.linalg.norm(array))
    # return the maximum cosine similarity
    return np.max(cos_sim)
    
def inverse_reward_func(array):
    if np.sum(array) == 1:
        return 1
    elif np.sum(array) == 0:
        return 0
    else:
        return 1/np.sum(array)

def span_reward_function(array):
    pass