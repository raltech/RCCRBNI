import numpy as np
from tqdm import tqdm

# UCB1 Agent
class UCB1Agent:
    def __init__(self, env, gamma, c, lr):
        self.env = env
        self.gamma = gamma
        self.c = c
        self.lr = lr
        self.Q = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.float32)
        self.n = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.uint32)
        self.done = False
    
    def choose_action(self):
        sum_n = np.sum(self.n[self.env.state])
        if sum_n == 0:
            action_idx = np.random.randint(0, self.env.n_elecs*self.env.n_amps)
        else:
            # exploration bonus
            bonus = self.c*np.sqrt(np.log(sum_n)/(self.n[self.env.state] + 1e-8))
            action_idx = np.argmax(self.Q[self.env.state]+bonus)
        return action_idx
    
    def update(self, state, action, reward, next_state):
        self.n[state, action] += 1
        self.Q[state, action] += self.lr*(reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state, action])

    def get_Q(self):
        return self.Q.copy()
    
    def get_n(self):
        return self.n.copy()
