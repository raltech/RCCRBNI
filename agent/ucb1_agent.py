import numpy as np
import math
from base_agent import Agent

class UCB1Agent(Agent):
    def __init__(self, action_space, learning_rate=0.01, discount_factor=0.99):
        super().__init__(action_space, learning_rate, discount_factor)
        self.n_actions = len(action_space)
        self.action_counts = np.zeros(self.n_actions)
        self.action_values = np.zeros(self.n_actions)

    def choose_action(self, _=None):
        total_counts = np.sum(self.action_counts)

        if total_counts == 0:
            # Choose a random action if no action has been taken yet
            return np.random.choice(self.n_actions)

        # Calculate UCB-1 values for all actions
        ucb_values = self.action_values + np.sqrt(2 * math.log(total_counts) / self.action_counts)

        # Choose the action with the highest UCB-1 value
        return np.argmax(ucb_values)

    def update(self, _, action, reward, __, ___):
        # Update the action counts and values
        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]

    def save(self, file_path):
        # Saving and loading not implemented for this example
        pass

    def load(self, file_path):
        # Saving and loading not implemented for this example
        pass
