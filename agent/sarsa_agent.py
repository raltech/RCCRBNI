import numpy as np

class SARSAAgent:
    def __init__(self, env, epsilon, gamma, lr):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
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
        next_action = self.choose_action()
        # self.n[state, action] += 1
        self.Q[state, action] += self.lr * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])

    def get_Q(self):
        return self.Q.copy()
    
    # def get_n(self):
    #     return self.n.copy()
