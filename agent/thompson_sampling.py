import numpy as np
from tqdm import tqdm

# Thompson Sampling Agent
class TSAgent:
    def __init__(self, env, gamma, lr):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.Q = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.float32)
        self.n = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.uint16)
        self.done = False
    
    def choose_action(self):
        prob_dist = np.log(np.copy(self.Q[self.env.state]) + 5.0)
        prob_dist_sum =np.sum(prob_dist)
        if prob_dist_sum == 0:
            action_idx = np.random.randint(0, self.env.n_elecs*self.env.n_amps)
        else:   
            prob_dist=prob_dist/prob_dist_sum
            action_idx = np.random.choice(len(self.Q[self.env.state]), size=1, p=prob_dist)
        return action_idx
    
    def update(self, state, action, reward, next_state):
        self.n[state, action] += 1
        # self.Q[self.state, action] = reward + self.gamma*np.max(self.Q[next_state])
        self.Q[state, action] += self.lr*(reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state, action])
    
    def get_Q(self):
        return self.Q.copy()
    
    def get_n(self):
        return self.n.copy()