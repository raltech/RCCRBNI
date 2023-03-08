import numpy as np

# Eplison Greedy Agent
'''
Epsilon Greedy Agent takes in a simulation environment and uses epsilon greedy policy to find the optimal policy
'''

class EpsilonGreedyAgent:
    def __init__(self, env, epsilon=0.1, gamma=0.9, alpha=0.1):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((env.n_cells, env.n_elecs*env.n_amps), dtype=np.float32)
        self.n = np.zeros((env.n_cells, env.n_elecs*env.n_amps), dtype=np.uint16)
        self.policy = np.zeros((env.n_cells), dtype=np.uint16)
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.action = self.policy[self.state]
        return self.state, self.action
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.env.n_elecs*self.env.n_amps)
        else:
            action_idx = np.argmax(self.Q[state])
            
        # convert action_idx to (elec, amp)
        action = (action_idx//self.env.n_amps + 1, action_idx%self.env.n_amps + 1)
        return action
    
    def step(self):
        self.action = self.get_action(self.state)
        self.s_next, self.reward,self.done = self.env.step(self.action)
        return self.s_next, self.reward, self.done
    
    def update(self):
        self.n[self.state, self.action] += 1
        self.Q[self.state, self.action] += self.alpha*(self.reward + self.gamma*np.max(self.Q[self.s_next]))
        self.policy[self.state] = np.argmax(self.Q[self.state])
        self.state = self.s_next
        return self.state, self.action
    
    def run(self, n_episodes=1000):
        self.reset()
        for i in range(n_episodes):
            # while not self.done:
            self.step()
            self.update()
        return self.policy
    
    def render(self):
        pass
