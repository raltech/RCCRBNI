import numpy as np

class Agent:
    def __init__(self, action_space, learning_rate=0.01, discount_factor=0.99):
        """
        Initialize the reinforcement learning agent.

        :param action_space: The action space of the environment.
        :param learning_rate: The learning rate used for updating the agent's knowledge.
        :param discount_factor: The discount factor for future rewards.
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def choose_action(self, state):
        """
        Choose an action based on the current state.

        :param state: The current state of the environment.
        :return: The chosen action.
        """
        raise NotImplementedError("This method needs to be implemented in a subclass.")

    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on the experience.

        :param state: The current state of the environment.
        :param action: The action taken by the agent.
        :param reward: The reward received after taking the action.
        :param next_state: The next state of the environment.
        :param done: Whether the episode has terminated.
        """
        raise NotImplementedError("This method needs to be implemented in a subclass.")

    def save(self, file_path):
        """
        Save the agent's knowledge to a file.

        :param file_path: The path of the file to save the agent's knowledge.
        """
        raise NotImplementedError("This method needs to be implemented in a subclass.")

    def load(self, file_path):
        """
        Load the agent's knowledge from a file.

        :param file_path: The path of the file containing the agent's knowledge.
        """
        raise NotImplementedError("This method needs to be implemented in a subclass.")
