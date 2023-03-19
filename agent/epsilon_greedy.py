import numpy as np
from tqdm import tqdm

# Eplison Greedy Agent
class EpsilonGreedyAgent:
    def __init__(self, env, epsilon, gamma, decay_rate, lr):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.lr = lr
        self.Q = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.float32)
        # self.n = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.uint32)
        self.done = False
    
    def choose_action(self):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.env.n_elecs*self.env.n_amps)
        else:
            action_idx = np.argmax(self.Q[self.env.state])
        return action_idx
    
    def update(self, state, action, reward, next_state):
        # self.n[state, action] += 1
        # self.Q[self.state, action] = reward + self.gamma*np.max(self.Q[next_state])
        self.Q[state, action] += self.lr*(reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state, action])
        self.epsilon *= self.decay_rate

    def get_Q(self):
        return self.Q.copy()
    
    # def get_n(self):
    #     return self.n.copy()
