import abc
import gym
from copy import copy
from typing import Optional
        
import numpy as np
from numpy.random import default_rng


from mbt_gym.gym.N_index_names import get_index_map

from mbt_gym.stochastic_processes.N_arrival_models import ArrivalModel
from mbt_gym.stochastic_processes.N_fill_probability_models import FillProbabilityModel
from mbt_gym.stochastic_processes.N_midprice_models import MidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel


class Bek_ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        price_impact_model: PriceImpactModel = None,
        lob_depth_model: ArrivalModel = None,  # Optional
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: int = None,
    ):
        self.num_trajectories = num_trajectories
        self.num_assets = num_assets
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.lob_depth_model = lob_depth_model
        self.rng = default_rng(seed)
        self.seed_ = seed

        self.fill_multiplier = self._get_fill_multiplier()  # Shape: (N, A, 2)
        self.round_initial_inventory = False
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)

        self.state = None  # Will be initialized in the TradingEnvironment

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass  # Overridden in subclasses

    def get_fills(self, action: np.ndarray):
        pass

    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None

    def _limit_depths(self, action: np.ndarray):
        # Expecting shape (N, A, 2), clip if needed in subclass
        return action[:, :, :2]

    def get_action_space(self) -> gym.spaces.Space:
        pass

    def get_required_stochastic_processes(self):
        return []

    def _get_max_depth(self) -> Optional[float]:
        if self.fill_probability_model is not None:
            return self.fill_probability_model.max_depth
        else:
            return None

    def _get_max_speed(self) -> float:
        if self.price_impact_model is not None:
            return self.price_impact_model.max_speed
        else:
            return None

    def _get_fill_multiplier(self):
        """
        Fill multiplier to convert bid/ask fills into signed inventory/cash effects.
        Shape: (num_trajectories, num_assets, 2)
        Values: [-1, +1] for [bid, ask] per asset.
        """
        ones = np.ones((self.num_trajectories, self.num_assets, 1))
        return np.concatenate([-ones, ones], axis=2)

    def _check_processes_are_not_none(self, processes):
        for process in processes:
            self._check_process_is_not_none(process)

    def _check_process_is_not_none(self, process: str):
        assert getattr(self, process) is not None, f"env.{process} must not be None in this model dynamics."

    @property
    def midprice(self):
        """
        Returns: midprice per trajectory and asset.
        Shape: (num_trajectories, num_assets)
        """
        return self.midprice_model.current_state[:, :, 0]


class Bek_LimitOrderModelDynamics(Bek_ModelDynamics):
    """ModelDynamics for 'limit' orders in a multi-asset setting."""
    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        lob_depth_model: ArrivalModel = None,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: int = None,
        max_depth: float = None,
    ):
        super().__init__(
            midprice_model=midprice_model,
            arrival_model=arrival_model,
            fill_probability_model=fill_probability_model,
            lob_depth_model=lob_depth_model,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )
        self.max_depth = max_depth or self._get_max_depth()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        """
        Update state after market orders and fills.
    
        arrivals: shape (N, A, 2)
        fills:    shape (N, A, 2)
        action:   shape (N, A, 2)
        """
        idx = get_index_map(self.num_assets)
    
        # === Update per-asset inventory ===
        # signed fill (buy = +1, sell = -1)
        signed_fills = arrivals * fills * -self.fill_multiplier  # shape (N, A, 2)
        inventory_change = np.sum(signed_fills, axis=2)  # sum over bid/ask → shape (N, A)
    
        # Add to the inventory slice in state
        self.state[:, idx["INVENTORY_INDEX"]:idx["TIME_INDEX"]] += inventory_change
    
        # === Update cash ===
        midprice_expanded = self.midprice[:, :, None]  # shape (N, A, 1)
        posting_depths = self._limit_depths(action)    # shape (N, A, 2)
        execution_prices = midprice_expanded + posting_depths * self.fill_multiplier  # shape (N, A, 2)
    
        cash_change = np.sum(self.fill_multiplier * arrivals * fills * execution_prices, axis=(1, 2))  # shape (N,)
        self.state[:, idx["CASH_INDEX"]] += cash_change


    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders, max_depth cannot be None."
        return gym.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(self.max_depth),
            shape=(self.num_assets, 2),  # One [bid, ask] per asset
        )

    def get_required_stochastic_processes(self):
        req = ["arrival_model", "fill_probability_model"]
    
        # Ask the fill model whether it needs extra processes (e.g. lob_depth_model)
        if self.fill_probability_model is not None:
            extra = getattr(self.fill_probability_model, "required_processes", None)
            if callable(extra):
                req.extend(extra())
    
        # remove duplicates while preserving order
        seen = set()
        req_unique = []
        for x in req:
            if x not in seen:
                seen.add(x)
                req_unique.append(x)
        return req_unique



    def get_arrivals_and_fills(self, action: np.ndarray):
        """
        Compute new arrivals and fills based on current intensities and depths.

        Args:
            action: shape (N, A, 2)
        Returns:
            arrivals: (N, A, 2)
            fills: (N, A, 2)
        """
        arrivals = self.arrival_model.get_arrivals()
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths)
        return arrivals, fills


