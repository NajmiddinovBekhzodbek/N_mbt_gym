import abc
from typing import Union, List

import gym
import numpy as np


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, done: bool) -> dict:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass


class ActionInfoCalculator(InfoCalculator):
    """ActionInfoCalculator records the actions taken throughout the episode and then outputs the mean actions taken at
    the terminal step as an info dict. This is the Stable Baselines 3 convention. See for example, the VecMonitor class
    of SB3."""

    def __init__(self, action_space: gym.spaces.Box, n_steps: int = 10 * 10, num_trajectories: int = 1000):
        self.action_space = action_space
        self.n_steps = n_steps
        self.num_trajectories = num_trajectories
        # action_space.shape[0] simply returns the number of actions
        self.nan_matrix = np.empty((self.num_trajectories, self.action_space.shape[0], self.n_steps))
        self.nan_matrix[:] = np.nan
        self.actions = self.nan_matrix.copy() # at the beginning actios are empty
        # {} means an empty dictionary. So creating a {} one for each Trajectory
        # Each dictionary can later be used to store custom information 
        self.empty_infos = [{} for _ in range(self.num_trajectories)] if self.num_trajectories > 1 else {}
        self.count = 0

    def calculate(
        self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, done: bool
    ) -> Union[dict, List[dict]]:
        if done:
            # for each trajectory and each action type over n_steps 
            mean_actions = self._calculate_mean_actions()
            # i means trajectory index and j is action type
            # creating a list of dictionaries. It will look like {"action_0": 0.5, "action_1": 1.2}
            # something like {"action_0": 0.5, "action_1": 1.2} will be created for each trajectory
            # mean_actions.shape[0] returns number of trajectories
            return [
                {f"action_{j}": mean_actions[i, j] for j in range(mean_actions.shape[1])}
                for i in range(mean_actions.shape[0])
            ]
        else:
            # action is the action taken by the agent at the current timestep. 
            # Its shape is (num_trajectories, action_space.shape[0]).
            # So, store the current action in the self.actions array 
            # at the index corresponding to the current timestep (self.count).
            # at t=1, means will be returned a dictionary
            self.actions[:, :, self.count] = action
            self.count += 1
            return self.empty_infos

    def reset(self, initial_state: np.ndarray):
        self.count = 0
        self.actions = self.nan_matrix.copy()

    # This computes the mean action across all time steps (n_steps) for each trajectory 
    #produces an array of shape (self.num_trajectories, self.action_space.shape[0])
    # where self.action_space.shape[0] means number of actions
    def _calculate_mean_actions(self):
        return self.actions.nanmean(axis=2)
