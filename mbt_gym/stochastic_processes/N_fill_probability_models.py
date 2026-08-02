from __future__ import annotations
import abc
from typing import Optional, Tuple, Union

import numpy as np

from mbt_gym.stochastic_processes.N_StochasticProcessModel import StochasticProcessModel

class FillProbabilityModel(StochasticProcessModel):
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

    # dependency contract (default: nothing extra required)
    def required_processes(self) -> list[str]:
        """
        Return names of extra stochastic processes required by this fill model,
        e.g. ["lob_depth_model"].

        Most fill models require nothing extra.
        """
        return []

    @abc.abstractmethod
    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        """
        depths: shape (num_trajectories, num_assets, 2)
        returns: fill_probabilities of same shape
        """
        raise NotImplementedError

    def get_fills(self, depths: np.ndarray) -> np.ndarray:
        assert depths.shape == (self.num_trajectories, self.num_assets, 2), \
            (
                f"Expected depths shape ({self.num_trajectories}, {self.num_assets}, 2), "
                f"got {depths.shape}"
            )

        unif = self.rng.uniform(size=(self.num_trajectories, self.num_assets, 2))
        fill_probs = self._get_fill_probabilities(depths)  # (N, A, 2)

        assert fill_probs.shape == (self.num_trajectories, self.num_assets, 2), \
            (
                f"_get_fill_probabilities must return shape "
                f"({self.num_trajectories}, {self.num_assets}, 2), got {fill_probs.shape}"
            )

        return unif < fill_probs

    @property
    @abc.abstractmethod
    def max_depth(self) -> float:
        raise NotImplementedError



class ExponentialFillFunction(FillProbabilityModel):
    """
    Multi-asset exponential fill model.

    p = exp(-k * depth)

    This model is memoryless, but N_StochasticProcessModel requires
    initial_state to be 3D: (num_trajectories, num_assets, state_dim).
    So we store a dummy 1D state per asset (state_dim=1).
    """

    def __init__(
        self,
        fill_exponent: Union[float, np.ndarray] = 1.5,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        if self.num_assets < 1:
            raise ValueError(f"num_assets must be >= 1, got {num_assets}")

        fe = np.array(fill_exponent, dtype=float)
        if fe.ndim == 0:
            self.fill_exponent = np.full((self.num_assets,), float(fe), dtype=float)
        elif fe.ndim == 1 and fe.shape[0] == self.num_assets:
            self.fill_exponent = fe.astype(float)
        else:
            raise ValueError(
                f"fill_exponent must be a scalar or shape ({self.num_assets},), got {fe.shape}"
            )

        # N-style: dummy state dim = 1 per asset
        dummy_state_dim = 1
        min_value = np.full((self.num_assets, dummy_state_dim), -np.inf, dtype=float)
        max_value = np.full((self.num_assets, dummy_state_dim),  np.inf, dtype=float)
        initial_state = np.zeros((int(num_trajectories), self.num_assets, dummy_state_dim), dtype=float)

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=float(step_size),
            terminal_time=0.0,  # not used
            initial_state=initial_state,
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    # declares extra processes required by this fill model
    def required_processes(self) -> list[str]:
        # This fill model does NOT depend on LOB depth state process.
        return []

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        depths = np.asarray(depths, dtype=float)

        # Expand scalar/asset vector k to broadcast over (M,A,2)
        k = self.fill_exponent.reshape(1, self.num_assets, 1)

        # Allow legacy N=1 shape (M,2)
        if depths.ndim == 2 and depths.shape[-1] == 2 and self.num_assets == 1:
            depths = depths[:, None, :]

        if depths.ndim != 3 or depths.shape[1] != self.num_assets or depths.shape[2] != 2:
            raise ValueError(
                f"Expected depths shape (M,{self.num_assets},2) (or (M,2) for N=1), got {depths.shape}"
            )

        probs = np.exp(-k * depths)
        return np.clip(probs, 0.0, 1.0)

    @property
    def max_depth(self) -> float:
        k_min = float(np.min(self.fill_exponent))
        if k_min <= 0:
            raise ValueError(f"fill_exponent must be positive, got min={k_min}")
        return -np.log(0.01) / k_min

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass


# Sort of linear decay rather than exponential I guess
class TriangularFillFunction(FillProbabilityModel):
    """
    Multi-asset triangular (piecewise-linear) fill model.

    For each trajectory and asset:
        d_max = max(depth_bid, depth_ask)
        p = max(1 - d_max / max_fill_depth, 0)

    Then we apply the same p to both bid/ask sides (because the model is based on max depth).

    Notes:
    - Memoryless model, but N_StochasticProcessModel requires initial_state to be 3D:
        (num_trajectories, num_assets, state_dim)
      so we store a dummy 1D state per asset (state_dim=1).
    - Accepts depths shaped (M, A, 2) or legacy (M, 2) when num_assets == 1.
    """

    def __init__(
        self,
        max_fill_depth: Union[float, np.ndarray] = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        if self.num_assets < 1:
            raise ValueError(f"num_assets must be >= 1, got {num_assets}")

        mfd = np.array(max_fill_depth, dtype=float)
        if mfd.ndim == 0:
            self.max_fill_depth = np.full((self.num_assets,), float(mfd), dtype=float)
        elif mfd.ndim == 1 and mfd.shape[0] == self.num_assets:
            self.max_fill_depth = mfd.astype(float)
        else:
            raise ValueError(
                f"max_fill_depth must be a scalar or shape ({self.num_assets},), got {mfd.shape}"
            )

        if np.any(self.max_fill_depth <= 0):
            raise ValueError(f"max_fill_depth must be > 0 for all assets, got {self.max_fill_depth}")

        # N-style: dummy state dim = 1 per asset
        dummy_state_dim = 1
        min_value = np.full((self.num_assets, dummy_state_dim), -np.inf, dtype=float)
        max_value = np.full((self.num_assets, dummy_state_dim),  np.inf, dtype=float)
        initial_state = np.zeros((int(num_trajectories), self.num_assets, dummy_state_dim), dtype=float)

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=float(step_size),
            terminal_time=0.0,  # Probably set to 0 because it is irrelevant (not used)
            initial_state=initial_state,
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    # declares extra processes required by this fill model
    def required_processes(self) -> list[str]:
        # This fill model does NOT depend on LOB depth state process.
        return []

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        depths = np.asarray(depths, dtype=float)

        # Allow legacy N=1 shape (M,2)
        if depths.ndim == 2 and depths.shape[-1] == 2 and self.num_assets == 1:
            depths = depths[:, None, :]

        if depths.ndim != 3 or depths.shape[1] != self.num_assets or depths.shape[2] != 2:
            raise ValueError(
                f"Expected depths shape (M,{self.num_assets},2) (or (M,2) for N=1), got {depths.shape}"
            )

        # Per-asset max_fill_depth (A,) -> (1,A,1) for broadcasting
        mfd = self.max_fill_depth.reshape(1, self.num_assets, 1)

        # d_max over bid/ask -> (M,A,1)
        d_max = np.max(depths, axis=2, keepdims=True)

        # p = max(1 - d_max / max_fill_depth, 0)
        p = 1.0 - (d_max / mfd)
        p = np.clip(p, 0.0, 1.0)  # (M,A,1)

        # Apply same p to both sides -> (M,A,2)
        probs = np.repeat(p, repeats=2, axis=2)
        return probs

    @property
    def max_depth(self) -> float:
        # very important quantity. Directly determines the range of allowable actions in Bek_ModelDynamics
        # Use the largest asset max_fill_depth for a safe global action bound.
        return 1.5 * float(np.max(self.max_fill_depth))

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass



class PowerFillFunction(FillProbabilityModel):
    """
    Multi-asset power (a.k.a. rational) fill model.

    Per asset i and side s in {bid, ask}:
        p_i,s(depth) = 1 / ( 1 + (m_i * depth_i,s)^(k_i) )

    Notes:
    - Memoryless model (no real state), but N_StochasticProcessModel requires
      initial_state to be 3D: (num_trajectories, num_assets, state_dim).
      We keep a dummy state_dim=1 per asset.
    - Expects depths shape (M, A, 2). For legacy single-asset you may pass (M, 2).
    """

    def __init__(
        self,
        fill_exponent: Union[float, np.ndarray] = 1.5,
        fill_multiplier: Union[float, np.ndarray] = 1.5,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        if self.num_assets < 1:
            raise ValueError(f"num_assets must be >= 1, got {num_assets}")

        # ---- fill_exponent: scalar or (A,) ----
        k = np.array(fill_exponent, dtype=float)
        if k.ndim == 0:
            self.fill_exponent = np.full((self.num_assets,), float(k), dtype=float)
        elif k.ndim == 1 and k.shape[0] == self.num_assets:
            self.fill_exponent = k.astype(float)
        else:
            raise ValueError(
                f"fill_exponent must be a scalar or shape ({self.num_assets},), got {k.shape}"
            )

        # ---- fill_multiplier: scalar or (A,) ----
        m = np.array(fill_multiplier, dtype=float)
        if m.ndim == 0:
            self.fill_multiplier = np.full((self.num_assets,), float(m), dtype=float)
        elif m.ndim == 1 and m.shape[0] == self.num_assets:
            self.fill_multiplier = m.astype(float)
        else:
            raise ValueError(
                f"fill_multiplier must be a scalar or shape ({self.num_assets},), got {m.shape}"
            )

        if np.any(self.fill_exponent <= 0):
            raise ValueError(f"fill_exponent must be > 0 for all assets, got {self.fill_exponent}")
        if np.any(self.fill_multiplier <= 0):
            raise ValueError(f"fill_multiplier must be > 0 for all assets, got {self.fill_multiplier}")

        # N-style: dummy state dim = 1 per asset
        dummy_state_dim = 1
        min_value = np.full((self.num_assets, dummy_state_dim), -np.inf, dtype=float)
        max_value = np.full((self.num_assets, dummy_state_dim),  np.inf, dtype=float)
        initial_state = np.zeros((int(num_trajectories), self.num_assets, dummy_state_dim), dtype=float)

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=float(step_size),
            terminal_time=0.0,  # not used
            initial_state=initial_state,
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    # declares extra processes required by this fill model
    def required_processes(self) -> list[str]:
        # This fill model does NOT depend on LOB depth state process.
        return []

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        depths = np.asarray(depths, dtype=float)

        # Allow legacy N=1 shape (M,2)
        if depths.ndim == 2 and depths.shape[-1] == 2 and self.num_assets == 1:
            depths = depths[:, None, :]

        if depths.ndim != 3 or depths.shape[1] != self.num_assets or depths.shape[2] != 2:
            raise ValueError(
                f"Expected depths shape (M,{self.num_assets},2) (or (M,2) for N=1), got {depths.shape}"
            )

        # Broadcast per-asset parameters over (M,A,2)
        k = self.fill_exponent.reshape(1, self.num_assets, 1)
        m = self.fill_multiplier.reshape(1, self.num_assets, 1)

        # Core model: p = (1 + (m*depth)^k)^(-1)
        x = np.maximum(m * depths, 0.0)  # depth should be nonnegative; guard anyway
        probs = 1.0 / (1.0 + np.power(x, k))

        return np.clip(probs, 0.0, 1.0)

    # very important quantity. Directly determines the range of allowable actions in Bek_ModelDynamics
    @property
    def max_depth(self) -> float:
        """
        Depth threshold corresponding to p(depth) = 0.01, using the most conservative asset:
            0.01 = 1 / (1 + (m d)^k)
            (m d)^k = 0.01^-1 - 1 = 99
            d = 99^(1/k) / m
        We return min over assets (tightest constraint).
        """
        target = (0.01 ** -1) - 1.0  # 99
        k = self.fill_exponent
        m = self.fill_multiplier

        d_per_asset = np.power(target, 1.0 / k) / m
        return float(np.min(d_per_asset))

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass


class ExogenousMmFillProbabilityModel(FillProbabilityModel):
    """
    Multi-asset compatible version of the original ExogenousMmFillProbabilityModel.

    Logic is unchanged:
    - current_state stores the exogenous "best depth" thresholds
    - If quoted depth <= best depth: fill prob = 1
    - If quoted depth >  best depth: fill prob = base_fill_probability * exp(-k * (depth - best_depth))

    IMPORTANT:
    - For multi-asset, we require exogenous_best_depth_processes to be a tuple of length 2:
        (best_bid_depth_process, best_ask_depth_process)
      where each process has state_dim >= 1 per asset, and we only use the FIRST state dimension
      as the "best depth" level.
    - Each process is expected to be multi-asset aware (num_assets matches this model).
    """

    def __init__(
        self,
        exogenous_best_depth_processes: Tuple[StochasticProcessModel, StochasticProcessModel],
        fill_exponent: float = 1.5,
        base_fill_probability: float = 1.0,
        step_size: float = 0.1,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        assert (
            len(exogenous_best_depth_processes) == 2
        ), "exogenous_best_depth_processes must be length 2 (bid and ask)"

        assert all(
            getattr(process, "initial_state", None) is not None and process.initial_state.shape[-1] > 0
            for process in exogenous_best_depth_processes
        ), "Exogenous best depth processes must have a state of at least size 1."

        self.num_assets = int(num_assets)
        self.exogenous_best_depth_processes = exogenous_best_depth_processes
        self.fill_exponent = float(fill_exponent)
        self.base_fill_probability = float(base_fill_probability)

        # ---- Validate process compatibility + extract initial_state/min/max in N-style ----
        bid_proc, ask_proc = self.exogenous_best_depth_processes

        for side_name, proc in [("bid", bid_proc), ("ask", ask_proc)]:
            if getattr(proc, "num_assets", None) is not None and int(proc.num_assets) != self.num_assets:
                raise ValueError(
                    f"Exogenous {side_name} depth process num_assets={proc.num_assets} "
                    f"does not match model num_assets={self.num_assets}."
                )
            if proc.initial_state.shape[0] != int(num_trajectories):
                raise ValueError(
                    f"Exogenous {side_name} depth process initial_state first dim must be "
                    f"num_trajectories={num_trajectories}, got {proc.initial_state.shape[0]}."
                )
            if proc.initial_state.shape[1] != self.num_assets:
                raise ValueError(
                    f"Exogenous {side_name} depth process initial_state second dim must be "
                    f"num_assets={self.num_assets}, got {proc.initial_state.shape[1]}."
                )

        # We use ONLY the first state dimension as the "best depth" level:
        # bid_best: (N,A,1), ask_best: (N,A,1)
        bid_best = bid_proc.initial_state[:, :, 0:1]
        ask_best = ask_proc.initial_state[:, :, 0:1]

        # Concatenate into this model's state: (N,A,2) = [best_bid_depth, best_ask_depth]
        initial_state = np.concatenate([bid_best, ask_best], axis=2)

        # min/max for this model state (A,2): take first dim from each process
        min_value = np.concatenate([bid_proc.min_value[:, 0:1], ask_proc.min_value[:, 0:1]], axis=1)
        max_value = np.concatenate([bid_proc.max_value[:, 0:1], ask_proc.max_value[:, 0:1]], axis=1)

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            terminal_time=0.0,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            num_assets=self.num_assets,
            seed=seed,
        )

    # declares extra processes required by this fill model
    def required_processes(self) -> list[str]:
        # This model depends on external best-depth processes (not named in env by default).
        # We keep this empty to avoid breaking your env builder logic.
        return []

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        """
        depths: (num_trajectories, num_assets, 2) -> (bid_depth, ask_depth)

        current_state: (num_trajectories, num_assets, 2) -> (best_bid_depth, best_ask_depth)
        """
        depths = np.asarray(depths, dtype=float)

        # Allow legacy N=1 shape (N,2)
        if depths.ndim == 2 and depths.shape[-1] == 2 and self.num_assets == 1:
            depths = depths[:, None, :]

        if depths.shape != (self.num_trajectories, self.num_assets, 2):
            raise ValueError(
                f"Expected depths shape ({self.num_trajectories},{self.num_assets},2), got {depths.shape}"
            )

        # Logic unchanged from original, just vectorized for (N,A,2)
        return (depths > self.current_state) * self.base_fill_probability * np.exp(
            -self.fill_exponent * (depths - self.current_state)
        ) + (depths <= self.current_state)

    @property
    def max_depth(self) -> float:
        # Same logic as original:
        # -log(0.01)/k + max(best_depth_max)
        # In multi-asset, take max across assets of the bid-process max_value (first dim).
        bid_proc = self.exogenous_best_depth_processes[0]
        best_depth_max = float(np.max(bid_proc.max_value[:, 0]))
        return -np.log(0.01) / self.fill_exponent + best_depth_max

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        # Update the exogenous processes first (logic unchanged)
        for process in self.exogenous_best_depth_processes:
            process.update(arrivals, fills, actions)

        # Refresh this model's current_state from the exogenous processes (multi-asset compatible)
        bid_proc, ask_proc = self.exogenous_best_depth_processes
        bid_best = bid_proc.current_state[:, :, 0:1]
        ask_best = ask_proc.current_state[:, :, 0:1]
        self.current_state = np.concatenate([bid_best, ask_best], axis=2)



class DynamicLOBExponentialFillFunction(FillProbabilityModel):
    """
    Computes fill probabilities using dynamic LOB depths instead of a constant fill exponent.

    IMPORTANT:
    - This model is *stateless* (state_dim = 0) to match the original single-asset behaviour.
    - It does NOT add extra columns to the environment state, for any num_assets.
    """
    def __init__(
        self,
        lob_depth_model: SynchronousLOBDepthModel,
        step_size: float = 0.1,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.lob_depth_model = lob_depth_model  # The dynamic LOB depth model

        # Make the model STATELESS: state_dim = 0
        state_dim = 0
        min_value = np.zeros((num_assets, state_dim))
        max_value = np.zeros((num_assets, state_dim))
        initial_state = np.zeros((num_trajectories, num_assets, state_dim))

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            terminal_time=0.0,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )

    # declares extra processes required by this fill model
    def required_processes(self) -> list[str]:
        return ["lob_depth_model"]

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        """
        Calculate fill probabilities using dynamic LOB depths.
        Args:
            depths: np.ndarray, the agent's action distances (shape: num_trajectories x num_assets x 2)
        Returns:
            Fill probabilities for bid and ask sides (same shape).
        """
        lob_depths = self.lob_depth_model.get_depths()  # Shape: (num_trajectories, num_assets, 2)
        assert lob_depths.shape == depths.shape, f"Shape mismatch: lob_depths {lob_depths.shape}, depths {depths.shape}"
        return np.exp(-lob_depths * depths)

    @property
    def max_depth(self) -> float:
        """
        Estimate max actionable depth using the smallest baseline depth.
        """
        assert self.lob_depth_model is not None, "LOB depth model is not defined."
        assert hasattr(self.lob_depth_model, 'c_baseline_depth'), "LOB depth model must have 'c_baseline_depth'."
        
        c_baseline_depth = self.lob_depth_model.c_baseline_depth  # Shape: (num_assets, 2)
        assert np.all(c_baseline_depth > 0), "All baseline depths must be positive."
        
        # Conservative (smallest baseline over all assets/sides)
        return np.log(0.01) / -np.min(c_baseline_depth)

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        """
        No-op: LOB depth model should be updated independently in the environment.
        """
        pass
