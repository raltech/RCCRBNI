import numpy as np
from tqdm import tqdm

# Thompson Sampling Agent
class TSAgent:
    def __init__(self, env, gamma=0.9, lr=0.1):
        self.state = 0
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.Q = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.float32)
        self.n = np.zeros((env.n_states, env.n_elecs*env.n_amps), dtype=np.uint16)
        self.reset()

    def reset(self):
        self.done = False
        self.state = self.env.reset()
        self.action = self.get_action()
        return self.state, self.action
    
    def get_action(self):
        prob_dist = np.log(np.copy(self.Q[self.state]) + 5.0)
        prob_dist_sum =np.sum(prob_dist)
        if prob_dist_sum == 0:
            action_idx = np.random.randint(0, self.env.n_elecs*self.env.n_amps)
        else:   
            prob_dist=prob_dist/prob_dist_sum
            action_idx = np.random.choice(len(self.Q[self.state]), size=1, p=prob_dist)
        return action_idx

    def map_action(self, action_idx):
        # convert action_idx to (elec, amp)
        action = (action_idx//self.env.n_amps + 1, action_idx%self.env.n_amps + 1)
        return action
    
    def step(self):
        self.action = self.get_action()
        self.next_state, self.reward, self.done = self.env.step(self.map_action(self.action))
    
    def update(self):
        self.n[self.state, self.action] += 1
        # self.Q[self.state, self.action] = self.reward + self.gamma*np.max(self.Q[self.next_state])
        self.Q[self.state, self.action] += self.lr*(self.reward + self.gamma*np.max(self.Q[self.next_state]) - self.Q[self.state, self.action])
        self.state = self.next_state
    
    def run(self, n_episodes=10000, log_freq=10000):
        new_dicts = []
        self.state, _ = self.reset()
        for i in tqdm(range(n_episodes)):
            if i % log_freq == 0 and i != 0:
                best_actions = np.nonzero(self.Q[self.state])[0]
                new_dicts.append(self.env.get_est_dictionary()[best_actions,:])
            if not self.done:
                self.step()
                self.update()
        return new_dicts
    
    def render(self):
        pass