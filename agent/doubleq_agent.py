import numpy as np

class DoubleQLearningAgent:
    def __init__(self, env, epsilon=0.8, gamma=0.9, decay_rate=1.0, lr=0.1):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.lr = lr
        self.Q1 = np.zeros((env.n_states, env.n_elecs * env.n_amps), dtype=np.float32)
        self.Q2 = np.zeros((env.n_states, env.n_elecs * env.n_amps), dtype=np.float32)
        self.n = np.zeros((env.n_states, env.n_elecs * env.n_amps), dtype=np.uint16)
        self.done = False

    def choose_action(self):
        Q_sum = self.Q1 + self.Q2
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.env.n_elecs * self.env.n_amps)
        else:
            action_idx = np.argmax(Q_sum[self.env.state])
        return action_idx

    def update(self, state, action, reward, next_state):
        self.n[state, action] += 1
        if np.random.random() < 0.5:
            best_action = np.argmax(self.Q1[next_state])
            self.Q1[state, action] += self.lr * (reward + self.gamma * self.Q2[next_state, best_action] - self.Q1[state, action])
        else:
            best_action = np.argmax(self.Q2[next_state])
            self.Q2[state, action] += self.lr * (reward + self.gamma * self.Q1[next_state, best_action] - self.Q2[state, action])
        self.epsilon *= self.decay_rate

    def get_Q(self):
        return self.Q1 + self.Q2

    def get_n(self):
        return self.n.copy()
