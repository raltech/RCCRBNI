import numpy as np
import os

'''
Simulation Class simulates the experiment environment for Reinforcement Learning agent
Arguments:
    - n_maxstep: the maximum number of steps the agent can take in the simulation
    - dictionary: each row represents a different electrode-amplitude pair. Each column represents a activation probability of different cell.
    - n_elecs: the number of electrodes
    - n_amps: the number of amplitudes
    - n_states: the number of states
    - elecs: the electrode numbers corresponding to the rows of the dictionary
    - amps: the amplitudes corresponding to the rows of the dictionary
    - elec_map: maps electrode numbers to their locations on the brain
    - cell_ids: the cell ids corresponding to the columns of the dictionary

Variables:
    - n_step: the current number of steps taken in the simulation
    - elec: the current electrode
    - amp: the current amplitude
    - done: whether the episode is done
    - reward: the reward for the current step
    - state: the current state of the simulation
    
Functions:
    - __init__: initializes the simulation
    - reset: resets the simulation to the initial state
    - step: takes in an action and returns the next state, reward, and whether the episode is done
    - sample: samples cell activations using the probability specified in the dictionary
    - render: renders the current state of the simulation
    - close: closes the simulation
'''
class SingleStateEnvironment:
    def __init__(self, path, reward_func, score_func, n_maxstep, n_elecs, n_amps, debug=False):
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
        self.score_func = score_func
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