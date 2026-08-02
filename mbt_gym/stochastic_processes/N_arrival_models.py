from __future__ import annotations
import abc
from typing import Optional, Union


import numpy as np

from mbt_gym.stochastic_processes.N_StochasticProcessModel import StochasticProcessModel
import abc

class ArrivalModel(StochasticProcessModel):  # Inherits from multi-asset-aware base class
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        num_assets: int = 1,  # For multiple assets
        seed: int = None,
    ):
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,  # Pass through
            seed=seed,
        )

    @abc.abstractmethod
    def get_arrivals(self) -> np.ndarray:
        """
        Returns the current intensities.
        Shape: (num_trajectories, 2 * num_assets) — 2 for bid/ask per asset.
        """
        pass



class PoissonArrivalModel(ArrivalModel):
    """
    Single-asset Poisson arrivals (memoryless, constant intensity).

    - Enforced: num_assets == 1
    - Output arrivals shape: (num_trajectories, 1, 2) for [bid, ask]
    - No internal state is stored (state dim = 0)
    """

    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        terminal_time: float = 0.0,  # kept for API compatibility; not used
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        if num_assets != 1:
            raise ValueError(
                f"PoissonArrivalModel supports num_assets==1 only, got num_assets={num_assets}. "
                "Use SynchronousHawkesArrivalModel (or another multi-asset arrival model) for num_assets>1."
            )

        self.intensity = np.array(intensity, dtype=float).reshape(1, 2)  # (1,2)

        # Memoryless -> zero-dimensional state, but keep batch axis consistent
        # current_state will be shape (N, 1, 0) and flatten to (N,0)
        empty_state = np.empty((num_trajectories, 1, 0), dtype=float)

        super().__init__(
            min_value=np.empty((1, 0), dtype=float),
            max_value=np.empty((1, 0), dtype=float),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=empty_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        # constant intensity -> no update needed
        return

    def get_arrivals(self) -> np.ndarray:
        # Bernoulli approx: P(arrival in dt) ≈ lambda * dt
        unif = self.rng.uniform(size=(self.num_trajectories, 1, 2))
        p = self.intensity.reshape(1, 1, 2) * self.step_size
        return (unif < p).astype(float)  # shape (N,1,2)


class PoissonArrivalNonLinearModel(ArrivalModel):
    """
    Single-asset Poisson arrivals (memoryless, constant intensity),
    but uses the exact Poisson probability over dt:

        P(arrival in dt) = 1 - exp(-lambda * dt)

    - Enforced: num_assets == 1
    - Output arrivals shape: (num_trajectories, 1, 2) for [bid, ask]
    - No internal state is stored (state dim = 0)
    """

    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        terminal_time: float = 0.0,  # kept for API compatibility; not used
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        if num_assets != 1:
            raise ValueError(
                f"PoissonArrivalNonLinearModel supports num_assets==1 only, got num_assets={num_assets}. "
                "Use SynchronousHawkesArrivalModel (or another multi-asset arrival model) for num_assets>1."
            )

        # shape: (1, 2) for [bid, ask]
        self.intensity = np.array(intensity, dtype=float).reshape(1, 2)

        # stateless -> state_dim = 0 but keep (T, A, state_dim)
        empty_state = np.empty((num_trajectories, 1, 0), dtype=float)

        super().__init__(
            min_value=np.empty((1, 0), dtype=float),
            max_value=np.empty((1, 0), dtype=float),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=empty_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        # constant intensity -> no state update
        return

    def get_arrivals(self) -> np.ndarray:
        # Exact Poisson event probability over dt:
        # p = 1 - exp(-lambda * dt)
        unif = self.rng.uniform(size=(self.num_trajectories, 1, 2))
        p = 1.0 - np.exp(-self.intensity.reshape(1, 1, 2) * self.step_size)
        return (unif < p).astype(float)  # shape (N, 1, 2)



class HawkesArrivalModel(ArrivalModel):
    """
    Multi-asset-aware *wrapper* of the original single-asset HawkesArrivalModel.

    IMPORTANT:
    - This implementation is intentionally restricted to num_assets == 1
      (same spirit as your N_PoissonArrivalModel).
    - Keeps the same update logic as the original HawkesArrivalModel:
        lambda_{t+dt} = lambda_t
                        + kappa * (baseline - lambda_t) * dt
                        + jump_size * arrivals
    - Uses Bernoulli thinning: arrivals ~ 1{U < lambda * dt}

    Shapes:
    - current_state: (num_trajectories, 1, 2)
    - get_arrivals(): (num_trajectories, 1, 2)  (bid, ask)
    - update() expects arrivals: (num_trajectories, 1, 2)
    """

    def __init__(
        self,
        baseline_arrival_rate: Union[np.ndarray, list] = np.array([[10.0, 10.0]]),
        step_size: float = 0.01,
        jump_size: float = 40.0,
        mean_reversion_speed: float = 60.0,
        terminal_time: float = 1.0,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        if num_assets != 1:
            raise ValueError(
                f"HawkesArrivalModel supports num_assets==1 only, got num_assets={num_assets}. "
                "Use SynchronousHawkesArrivalModel for multi-asset Hawkes dynamics."
            )

        self.num_assets = 1
        self.step_size = float(step_size)
        self.terminal_time = float(terminal_time)

        # Keep same semantics as original:
        # baseline_arrival_rate default was shape (1,2)
        baseline = np.array(baseline_arrival_rate, dtype=float)
        if baseline.shape == (2,):
            baseline = baseline.reshape(1, 2)
        if baseline.shape != (1, 2):
            raise ValueError(f"baseline_arrival_rate must be shape (2,) or (1,2), got {baseline.shape}")

        self.baseline_arrival_rate = baseline  # (1,2)
        self.jump_size = float(jump_size)
        self.mean_reversion_speed = float(mean_reversion_speed)

        # State is the intensity lambda (bid, ask) for the single asset
        initial_state = np.tile(baseline[None, :, :], (num_trajectories, 1, 1))  # (N,1,2)

        min_value = np.array([[0.0, 0.0]], dtype=float)  # (1,2)
        max_value = self._get_max_arrival_rate().astype(float)  # (1,2)

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=self.step_size,
            terminal_time=self.terminal_time,
            initial_state=initial_state,
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        action: np.ndarray,
        state: np.ndarray = None,
    ) -> None:
        """
        arrivals: expected shape (num_trajectories, 1, 2)
        """
        if arrivals.shape != (self.num_trajectories, 1, 2):
            raise ValueError(
                f"Expected arrivals shape ({self.num_trajectories}, 1, 2), got {arrivals.shape}"
            )

        # baseline: (1,2) -> broadcast to (N,1,2)
        baseline = self.baseline_arrival_rate.reshape(1, 1, 2)

        # Same logic as original HawkesArrivalModel, just with (N,1,2) tensors
        self.current_state = (
            self.current_state
            + self.mean_reversion_speed * (baseline - self.current_state) * self.step_size
            + self.jump_size * arrivals
        )

        # Optional safety clip (original didn’t explicitly, but it’s harmless and matches your style)
        self.current_state = np.clip(self.current_state, self.min_value.reshape(1, 1, 2), self.max_value.reshape(1, 1, 2))

    def get_arrivals(self) -> np.ndarray:
        """
        Sample arrivals using Bernoulli thinning:
            P(arrival in dt) = lambda * dt
        Output: (num_trajectories, 1, 2)
        """
        unif = self.rng.uniform(size=(self.num_trajectories, 1, 2))
        return (unif < self.current_state * self.step_size)

    def _get_max_arrival_rate(self) -> np.ndarray:
        # Keep same heuristic as original: baseline * 10
        return self.baseline_arrival_rate * 10.0


class SynchronousHawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        step_size: float,
        baseline_arrival_rate: Optional[np.ndarray] = None,
        mean_reversion_speed: Union[float, np.ndarray] = 70.0,
        self_jump_size: Union[float, np.ndarray] = 15.0,
        mutual_jump_size: Union[float, np.ndarray] = 10.0,
        synchrony_factor: Union[float, np.ndarray] = 0.2,
        cross_asset_influence: Union[float, np.ndarray] = 0.2,  # Cross-asset coupling
        # very important. Diagonals must be ignored and does not need to be symmetric
        num_trajectories: int = 1,
        terminal_time: float = 1.0,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = num_assets
        self.step_size = step_size
        self.terminal_time = terminal_time

        if baseline_arrival_rate is None:
            baseline = np.array([10.0, 10.0])  # Ensure float dtype
        else:
            baseline = np.array(baseline_arrival_rate, dtype=np.float64)


        if baseline.shape == (2,):
            baseline = np.tile(baseline, (num_assets, 1)).astype(np.float64)
        assert baseline.shape == (num_assets, 2), "baseline must be shape (num_assets, 2)"

        # Flexible parameter expansion
        def _expand_param(param, name, shape):
            param = np.array(param)
            if param.shape == ():  # scalar
                return np.full(shape, param)
            elif param.shape == (shape[0],):  # vector
                return np.tile(param.reshape(-1, 1), shape[1]) if len(shape) == 2 else param
            elif param.shape == shape:
                return param
            else:
                raise ValueError(f"{name} must be scalar, shape {shape}, or {shape[0]}-vector.")

        self.mean_reversion_speed = _expand_param(mean_reversion_speed, "mean_reversion_speed", (num_assets,))
        self.self_jump_size = _expand_param(self_jump_size, "self_jump_size", (num_assets,))
        self.mutual_jump_size = _expand_param(mutual_jump_size, "mutual_jump_size", (num_assets,))
        self.synchrony_factor = _expand_param(synchrony_factor, "synchrony_factor", (num_assets, ))
        self.cross_asset_influence = _expand_param(cross_asset_influence, "cross_asset_influence", (num_assets, num_assets))  # cross-asset
        self.baseline_arrival_rate = baseline_arrival_rate

        min_value = np.zeros_like(baseline, dtype=np.float64)
        max_value = baseline.astype(np.float64) * 10
        initial_state = np.tile(baseline[None, :, :], (num_trajectories, 1, 1))  # shape: (num_trajectories, num_assets, 2)



        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )
        # Now ensure correct shape (only after super().__init__)

    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: np.ndarray = None):
        """
        Multi-asset update of arrival intensities (λ).
        - arrivals: shape (num_trajectories, num_assets, 2)
        - current_state: shape (num_trajectories, num_assets, 2)
        """
    
        bid_index = 0
        ask_index = 1
        dt = self.step_size
    
        # --- OLD STATE (frozen) ---
        lambdas_old = self.current_state.copy()     # (num_trajectories, num_assets, 2)

        # --- state update (simultaneous) ---
        lambdas_new = lambdas_old.copy()

        baseline = self.initial_state[0]            # shape: (num_assets, 2)
    
        # ======================================================
        # === Intra-asset Hawkes dynamics (SIMULTANEOUS) =======
        # ======================================================
        lambdas_new[:, :, bid_index] += (
            self.mean_reversion_speed[None, :]
            * (
                baseline[:, bid_index]
                - lambdas_old[:, :, bid_index]
                + self.synchrony_factor[None, :] * lambdas_old[:, :, ask_index]
            )
            * dt
            + self.self_jump_size[None, :] * arrivals[:, :, bid_index]
            + self.mutual_jump_size[None, :] * arrivals[:, :, ask_index]
        )
    
        lambdas_new[:, :, ask_index] += (
            self.mean_reversion_speed[None, :]
            * (
                baseline[:, ask_index]
                - lambdas_old[:, :, ask_index]
                + self.synchrony_factor[None, :] * lambdas_old[:, :, bid_index]
            )
            * dt
            + self.self_jump_size[None, :] * arrivals[:, :, ask_index]
            + self.mutual_jump_size[None, :] * arrivals[:, :, bid_index]
        )
    
        # ======================================================
        # === Cross-asset excitation (SIMULTANEOUS) ============
        # ======================================================
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i == j:
                    continue
                lambdas_new[:, i, bid_index] += self.cross_asset_influence[i, j] * arrivals[:, j, bid_index]
                lambdas_new[:, i, ask_index] += self.cross_asset_influence[i, j] * arrivals[:, j, ask_index]
    
        # ======================================================
        # === Final clip ======================================
        # ======================================================
        # Be careful: clipping might also be applied elsewhere
        self.current_state = np.clip(lambdas_new, self.min_value, self.max_value)


    def get_arrivals(self) -> np.ndarray:
        """
        Sample actual bid/ask arrivals from current intensities using Bernoulli thinning.
        Output shape: (num_trajectories, num_assets, 2)
        """
        unif = self.rng.uniform(size=self.current_state.shape)  # shape: (num_trajectories, num_assets, 2)
        return (unif < self.current_state * self.step_size)

    def get_intensities(self) -> np.ndarray:
        """
        Returns the current MO arrival intensities for all assets and both sides.
        Shape: (num_trajectories, num_assets, 2)
        """
        return self.current_state



class SynchronousLOBDepthModel(ArrivalModel):
    def __init__(
        self,
        c_baseline_depth: Optional[np.ndarray] = None,
        c_mean_reversion_speed: Union[float, np.ndarray] = 50.0,
        c_self_jump_size: Union[float, np.ndarray] = 8.0,
        c_mutual_jump_size: Union[float, np.ndarray] = 5.0,
        c_synchrony_factor: Union[float, np.ndarray] = 0.1,
        c_cross_asset_influence: Union[float, np.ndarray] = 0.1,
        step_size: float = 0.01,
        terminal_time: float = 1.0,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = num_assets
        self.step_size = step_size
        self.terminal_time = terminal_time

        # Default baseline
        if c_baseline_depth is None:
            baseline = np.tile(np.array([3.0, 3.0]), (num_assets, 1))  # shape: (num_assets, 2)
        else:
            baseline = np.array(c_baseline_depth, dtype=np.float64)

        if baseline.shape == (2,):
            baseline = np.tile(baseline, (num_assets, 1)).astype(np.float64)
        assert baseline.shape == (num_assets, 2)

        def _expand(param, name, shape):
            param = np.array(param)
            if param.shape == ():  # scalar
                return np.full(shape, param)
            elif param.shape == (shape[0],):
                return np.tile(param.reshape(-1, 1), shape[1]) if len(shape) == 2 else param
            elif param.shape == shape:
                return param
            else:
                raise ValueError(f"{name} must be scalar, shape {shape}, or {shape[0]}-vector.")

        self.c_baseline_depth = baseline
        self.c_mean_reversion_speed = _expand(c_mean_reversion_speed, "c_mean_reversion_speed", (num_assets,))
        self.c_self_jump_size = _expand(c_self_jump_size, "c_self_jump_size", (num_assets,))
        self.c_mutual_jump_size = _expand(c_mutual_jump_size, "c_mutual_jump_size", (num_assets,))
        self.c_synchrony_factor = _expand(c_synchrony_factor, "c_synchrony_factor", (num_assets,))
        self.c_cross_asset_influence = _expand(c_cross_asset_influence, "c_cross_asset_influence", (num_assets, num_assets))

        baseline = baseline.astype(np.float64)  # Ensure float early on
        min_value = np.full_like(baseline, 1e-6)
        max_value = baseline * 10
        initial_state = np.tile(baseline[None, :, :], (num_trajectories, 1, 1))  # shape: (num_trajectories, num_assets, 2)




        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )
        
        # Now ensure correct shape (only after super().__init__)

    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, state: np.ndarray = None):
        """
        Multi-asset update of LOB depths (c).
        - arrivals: shape (num_trajectories, num_assets, 2)
        - current_state: shape (num_trajectories, num_assets, 2)
        """
    
        bid_index = 0
        ask_index = 1
        dt = self.step_size
    
        # --- OLD STATE (frozen) ---
        depths_old = self.current_state.copy()      # (num_trajectories, num_assets, 2)

        # --- state update (simultaneous) ---
        depths_new = depths_old.copy()

        baseline = self.initial_state[0]             # shape: (num_assets, 2)
    
        # ======================================================
        # === Intra-asset depth dynamics (SIMULTANEOUS) ========
        # ======================================================
        depths_new[:, :, bid_index] += (
            self.c_mean_reversion_speed[None, :]
            * (
                baseline[:, bid_index]
                - depths_old[:, :, bid_index]
                + self.c_synchrony_factor[None, :] * depths_old[:, :, ask_index]
            )
            * dt
            + self.c_self_jump_size[None, :] * arrivals[:, :, bid_index]
            + self.c_mutual_jump_size[None, :] * arrivals[:, :, ask_index]
        )
    
        depths_new[:, :, ask_index] += (
            self.c_mean_reversion_speed[None, :]
            * (
                baseline[:, ask_index]
                - depths_old[:, :, ask_index]
                + self.c_synchrony_factor[None, :] * depths_old[:, :, bid_index]
            )
            * dt
            + self.c_self_jump_size[None, :] * arrivals[:, :, ask_index]
            + self.c_mutual_jump_size[None, :] * arrivals[:, :, bid_index]
        )
    
        # ======================================================
        # === Cross-asset depth coupling (SIMULTANEOUS) ========
        # ======================================================
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i == j:
                    continue
                depths_new[:, i, bid_index] += self.c_cross_asset_influence[i, j] * arrivals[:, j, bid_index]
                depths_new[:, i, ask_index] += self.c_cross_asset_influence[i, j] * arrivals[:, j, ask_index]
    
        # ======================================================
        # === Final clip ======================================
        # ======================================================
        # Be careful: clipping might also be applied elsewhere
        self.current_state = np.clip(depths_new, self.min_value, self.max_value)

    def get_depths(self) -> np.ndarray:
        return self.current_state
        
    # becasue  SynchronousLOBDepthModel from ArrivalModel  ArrivalModel, it mus implement get_arrivals
    # we don't need get_arrivals here so we are using just a dummy function that returns zeros.
    # could be dangerous. Don't call this function
    def get_arrivals(self) -> np.ndarray:
        raise NotImplementedError("get_arrivals() is not applicable for LOB depths.")


