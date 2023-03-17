import numpy as np
from env.single_state_env import SingleStateEnvironment
from agent.epsilon_greedy import EpsilonGreedyAgent
from tqdm import tqdm

N_ELECTRODES = 512
N_AMPLITUDES = 42
N_EXAUSTIVE_SEARCH = N_ELECTRODES * N_AMPLITUDES * 25

def main():
    experiments = ["2022-11-04-2", "2022-11-28-1"]
    path = f"./data/{experiments[1]}/dictionary"

    env = SingleStateEnvironment(path, reward_func=inverse_reward_func, 
                                 n_maxstep=N_EXAUSTIVE_SEARCH, n_elecs=N_ELECTRODES, n_amps=N_AMPLITUDES)
    agent = EpsilonGreedyAgent(env, gamma=0.7, epsilon=0.8, decay_rate=1-10e-7, lr=0.1)
    # agent = TSAgent(env, gamma=0.9, lr=0.1)
    log_freq = 10000
    new_dicts = agent.run(n_episodes=50000, log_freq=log_freq)


if __name__ == "__main__":
    main()

class SingleStateEnvironment:
    def __init__(self, path, reward_func, n_maxstep, n_elecs, n_amps, debug=False):
        # Load relevant data from .npz files
        try:
            with np.load(os.path.join(path,"dictionary.npz")) as data:
                self.dict = data["dictionary"]
                self.elecs = data["entry_elecs"]
                self.amps = data["entry_amps"]
                self.elec_map = data["elec_map"]
            with np.load(os.path.join(path,"decoders.npz")) as data:
                self.cell_ids = data["cell_ids"]
        except FileNotFoundError:
            print("Please make sure the dictionary.npz and decoders.npz files are present in the specified path")

        # Initialize variables
        self.reward_func = reward_func
        self.n_maxstep = n_maxstep
        self.n_elecs = n_elecs
        self.n_amps = n_amps
        self.n_cells = len(self.cell_ids)
        self.n_states = 1
        self.debug = debug
        self.reset()
        
    def reset(self):
        # Reset variables
        self.n_step = 0
        self.elec = 0 # electrode number (1~n_elecs)
        self.amp = 0 # amplitude (1~n_amps)
        self.done = False
        self.reward = 0
        self.dict_hat = np.zeros((self.n_elecs*self.n_amps, len(self.cell_ids)), dtype=np.uint16)
        self.dict_hat_count = np.zeros((self.n_elecs*self.n_amps), dtype=np.uint16)
        self.state = 0
        return self.state
    
    def step(self, action):
        self.elec = action[0]
        self.amp = action[1]

        if self.n_step >= self.n_maxstep:
            self.done = True
        else:
            sampled_activations = self.sample(self.elec, self.amp)
            self.dict_hat[(self.elec-1)*self.n_amps + (self.amp-1)] += sampled_activations
            self.dict_hat_count[(self.elec-1)*self.n_amps + (self.amp-1)] += 1
            self.reward = self.reward_func(sampled_activations)
            self.n_step += 1
        return self.state, self.reward, self.done
    
    def sample(self, elec, amp):
        try:
            idx = np.where((self.elecs == elec) & (self.amps == amp))[0][0]
            dist = self.dict[idx]
        except IndexError:
            if self.debug: print(f"Electrode {elec} with amplitude {amp} was not in the dictionary")
            if self.debug: print(f"Assume no cells were activated")
            dist = np.zeros(len(self.cell_ids), dtype=np.float64)

        if np.any(dist < 0):
            invalid_idx = np.where(dist < 0)[0]
            if self.debug: print(f"Invalid value at index {invalid_idx}: {dist[invalid_idx]}")
            dist[invalid_idx] = 0
        if np.any(dist > 1):
            invalid_idx = np.where(dist > 1)[0]
            if self.debug: print(f"Invalid value at index {invalid_idx}: {dist[invalid_idx]}")
            dist[invalid_idx] = 1
        if np.any(np.isnan(dist)):
            invalid_idx = np.where(np.isnan(dist))[0]
            if self.debug: print(f"Invalid value at index {invalid_idx}: {dist[invalid_idx]}")
            dist[invalid_idx] = 0

        sampled_activations = np.random.binomial(1, dist).astype(dtype=np.uint8)

        return sampled_activations
    
    def render(self, elec, amp):
        print(self.dict_hat[(elec-1)*self.n_amps + (amp-1)])

    def get_est_dictionary(self):
        return self.dict_hat / self.dict_hat_count[:,np.newaxis]
    
    def close(self):
        pass

class EpsilonGreedyAgent:
    def __init__(self, env, epsilon=0.8, gamma=0.9, decay_rate=1.0, lr=0.1):
        self.state = 0
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
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
        self.next_state, self.reward, self.done = self.env.step(self.map_action(self.action))
    
    def update(self):
        # print(self.state, self.action, self.next_state)
        self.n[self.state, self.action] += 1
        # self.Q[self.state, self.action] = self.reward + self.gamma*np.max(self.Q[self.next_state])
        self.Q[self.state, self.action] += self.lr*(self.reward + self.gamma*np.max(self.Q[self.next_state]) - self.Q[self.state, self.action])
        self.state = self.next_state
        self.epsilon *= self.decay_rate
    
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