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
        num_assets: int = 1,
        seed: int = None,
    ):
        self.num_assets = num_assets
        self.num_trajectories = num_trajectories
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.terminal_time = terminal_time
        self.initial_state = initial_state  # shape: (num_trajectories, num_assets, state_dim_per_asset)
        self._check_attribute_shapes()

        self.current_state = copy(self.initial_vector_state)  # shape: (num_trajectories, num_assets, state_dim_per_asset)
        self.rng = default_rng(seed)
        self.seed_ = seed

    def reset(self):
        self.current_state = copy(self.initial_vector_state)

    def seed(self, seed: int = None):
        self.rng = default_rng(seed)
        self.seed_ = seed

    def _check_attribute_shapes(self):
        # Only check asset-dimension related shapes (num_assets x state_dim_per_asset)
        for name in ["min_value", "max_value"]:
            attr = getattr(self, name)
            assert attr.shape[0] == self.num_assets, f"{name} must have shape (num_assets, state_dim_per_asset)"

        # initial_state should be full (num_trajectories, num_assets, state_dim)
        assert len(self.initial_state.shape) == 3, \
            f"initial_state must be 3D: (num_trajectories, num_assets, state_dim), got {self.initial_state.shape}"
        assert self.initial_state.shape[0] == self.num_trajectories
        assert self.initial_state.shape[1] == self.num_assets

    @property
    def initial_vector_state(self) -> np.ndarray:
        """
        Return properly shaped initial state for current process.
        Shape: (num_trajectories, num_assets, state_dim)
        """
        return self.initial_state  # Already tiled in each subclass appropriately

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: np.ndarray = None):
        """
        Update the state based on arrivals/fills/actions.
        Each subclass must implement this method.
        """
        pass
