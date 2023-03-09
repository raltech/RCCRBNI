import numpy as np

# Eplison Greedy Agent
class EpsilonGreedyAgent:
    def __init__(self, env, epsilon=0.8, gamma=0.9, decay_rate=1.0):
        self.state = 0
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.Q = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.float32)
        self.n = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.uint16)
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.action = self.get_action()
        return self.state, self.action
    
    def get_action(self):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.env.n_elecs*self.env.n_amps)
        else:
            action_idx = np.argmax(self.Q[self.state])
        return action_idx

    def map_action(self, action_idx):
        # convert action_idx to (elec, amp)
        action = (action_idx//self.env.n_amps + 1, action_idx%self.env.n_amps + 1)
        return action
    
    def step(self):
        self.action = self.get_action()
        self.next_state, self.reward,self.done = self.env.step(self.map_action(self.action))
    
    def update(self):
        # print(self.state, self.action, self.next_state)
        self.n[self.state, self.action] += 1
        self.Q[self.state, self.action] = self.reward + self.gamma*np.max(self.Q[self.next_state, self.action])
        self.state = self.next_state
        self.epsilon *= self.decay_rate
    
    def run(self, n_episodes=1000):
        self.state, _ = self.reset()
        for i in range(n_episodes):
            # while not self.done:
            self.step()
            self.update()
        return self.Q
    
    def render(self):
        pass