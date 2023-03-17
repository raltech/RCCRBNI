import numpy as np
from base_agent import Agent

class DecayingEpsilonGreedyAgent(Agent):
    def __init__(self, action_space, learning_rate=0.01, discount_factor=0.99, epsilon_decay=0.999):
        super().__init__(action_space, learning_rate, discount_factor)
        self.n_actions = len(action_space)
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay

    def choose_action(self, _=None):
        if np.random.random() < self.epsilon:
            # Choose a random action with probability epsilon
            return np.random.choice(self.n_actions)
        else:
            # Choose the action with the highest Q-value
            return np.argmax(self.q_values)

    def update(self, _, action, reward, __, ___):
        # Update the Q-value for the chosen action
        self.action_counts[action] += 1
        self.q_values[action] += self.learning_rate * (reward - self.q_values[action])

        # Decay the epsilon value
        self.epsilon *= self.epsilon_decay

    def save(self, file_path):
        # Saving and loading not implemented for this example
        pass

    def load(self, file_path):
        # Saving and loading not implemented for this example
        pass
