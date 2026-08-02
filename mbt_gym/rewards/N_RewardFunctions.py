import abc
from typing import Union
import numpy as np
from mbt_gym.gym.N_index_names import get_index_map  # Use dynamic indexing based on num_assets


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal_step: bool = False,
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass


class PnL(RewardFunction):
    def __init__(self, num_assets: int = 1):
        self.num_assets = num_assets
        self.idx = get_index_map(num_assets)

    def calculate(self, current_state, action, next_state, is_terminal_step=False) -> np.ndarray:
        cash_change = next_state[:, self.idx["CASH_INDEX"]] - current_state[:, self.idx["CASH_INDEX"]]

        current_inventory = current_state[:, self.idx["INVENTORY_INDEX"]:self.idx["TIME_INDEX"]]  # (N, A)
        next_inventory = next_state[:, self.idx["INVENTORY_INDEX"]:self.idx["TIME_INDEX"]]        # (N, A)

        current_price = current_state[:, self.idx["ASSET_PRICE_INDEX"]:self.idx["ASSET_PRICE_INDEX"] + self.num_assets]
        next_price = next_state[:, self.idx["ASSET_PRICE_INDEX"]:self.idx["ASSET_PRICE_INDEX"] + self.num_assets]

        return cash_change + np.sum(next_inventory * next_price, axis=1) - np.sum(current_inventory * current_price, axis=1)

    def reset(self, initial_state: np.ndarray):
        pass



class CjMmCriterion(RewardFunction):
    """
    Multi-asset Cartea-Jaimungal Market Making criterion with:
    - Mark-to-market PnL
    - Running inventory penalty
    - Smooth decomposition of terminal inventory penalty
    """

    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
        terminal_time: float = 1.0,
        num_assets: int = 1,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.inventory_exponent = inventory_exponent
        self.terminal_time = terminal_time
        self.num_assets = num_assets

        self.idx = get_index_map(num_assets)
        self.pnl = PnL(num_assets=num_assets)

        self.initial_inventory = None
        self.episode_length = None

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal_step: bool = False,
    ) -> Union[float, np.ndarray]:
        dt = next_state[:, self.idx["TIME_INDEX"]] - current_state[:, self.idx["TIME_INDEX"]]

        # Extract inventory: shape (batch_size, num_assets)
        q_curr = current_state[:, self.idx["INVENTORY_INDEX"]:self.idx["TIME_INDEX"]]
        q_next = next_state[:, self.idx["INVENTORY_INDEX"]:self.idx["TIME_INDEX"]]
        q_init = self.initial_inventory  # shape (batch_size, num_assets)

        # Inventory power
        q_curr_pow = np.abs(q_curr) ** self.inventory_exponent
        q_next_pow = np.abs(q_next) ** self.inventory_exponent
        q_init_pow = np.abs(q_init) ** self.inventory_exponent

        # === Components ===
        pnl = self.pnl.calculate(current_state, action, next_state, is_terminal_step)

        running_penalty = dt * self.per_step_inventory_aversion * np.sum(q_next_pow, axis=1)

        terminal_penalty = self.terminal_inventory_aversion * (
            np.sum(q_next_pow - q_curr_pow, axis=1)
            + dt / self.episode_length * np.sum(q_init_pow, axis=1)
        )

        return pnl - running_penalty - terminal_penalty


    def reset(self, initial_state: np.ndarray):
        self.initial_inventory = initial_state[:, self.idx["INVENTORY_INDEX"]:self.idx["TIME_INDEX"]]
        self.episode_length = self.terminal_time - initial_state[:, self.idx["TIME_INDEX"]]
