import numpy as np

class MultiArmedBanditEnvironment:
    def __init__(self, n_arms, seed=None):
        """
        Initialize the multi-armed bandit environment.

        :param n_arms: The number of arms (actions) in the environment.
        :param seed: An optional seed for reproducibility.
        """
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.arm_probabilities = self.rng.uniform(size=n_arms)

    def step(self, action):
        """
        Take a step in the environment with the given action.

        :param action: The action to take (the arm to pull).
        :return: The reward for the action.
        """
        if 0 <= action < self.n_arms:
            reward = self.rng.binomial(n=1, p=self.arm_probabilities[action])
        else:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.n_arms - 1}.")

        return reward
