import abc
from copy import copy

import numpy as np
from numpy.random import default_rng


class StochasticProcessModel(metaclass=abc.ABCMeta):
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        # Force everything to float to avoid integer truncation
        self.min_value = np.array(min_value, dtype=np.float64)
        self.max_value = np.array(max_value, dtype=np.float64)
        self.step_size = float(step_size)
        self.terminal_time = float(terminal_time)
        self.num_trajectories = num_trajectories
        self.initial_state = np.array(initial_state, dtype=np.float64)

        self._check_attribute_shapes()  # makes sure that the shapes are correct
        self.current_state = copy(self.initial_vector_state)  # important attribute
        self.rng = default_rng(seed)  # random number generator
        self.seed_ = seed


    def reset(self):
        self.current_state = self.initial_vector_state # so all StochasticProcesses will inherit this method 

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: np.ndarray = None): # different across different processes
        pass
        
    # This method helps to reset the RNG even after an instsnce is created Ex >>> Create Instance : model = StochasticProcessModel(seed=123)
    # Re-seed with a new value: model.seed(456) 
    # So we can change the seed dynamically without having to recreate the entire object (whithout making a new instance)
    def seed(self, seed: int = None): 
        self.rng = default_rng(seed) 
        self.seed_ = seed 

    # The arrays must be 2 dimensional ([[]]). First dimension should be 1 but the second dimension is not restricted. 
    # This means they can be of shape (1,1) or (1,2) or even (1,3)! 
    def _check_attribute_shapes(self):
        for name in ["initial_state", "min_value", "max_value"]:
            attribute = getattr(self, name)
            assert (
                len(attribute.shape) == 2 and attribute.shape[0] == 1
            ), f"Attribute {name} must be a vector of shape (1, state_size)."

    @property
    def initial_vector_state(self) -> np.ndarray:
        initial_state = self.initial_state
        if isinstance(initial_state, list):
            initial_state = np.array([self.initial_state])
        return np.repeat(initial_state, self.num_trajectories, axis=0)
