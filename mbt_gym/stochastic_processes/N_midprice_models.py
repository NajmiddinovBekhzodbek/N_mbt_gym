from __future__ import annotations
from math import sqrt
from typing import Optional, Union  # Ensure Union is imported


import numpy as np

from mbt_gym.stochastic_processes.N_StochasticProcessModel import StochasticProcessModel

MidpriceModel = StochasticProcessModel

from mbt_gym.gym.index_names import BID_INDEX, ASK_INDEX


class ConstantMidpriceModel(MidpriceModel):
    """
    Multi-asset constant midprice model.

    State per asset: 1D (the midprice)
    Shapes:
      - initial_state: (num_trajectories, num_assets, 1)
      - min_value/max_value: (num_assets, 1)
    """

    def __init__(
        self,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.terminal_time = float(terminal_time)

        # --- Force initial_price to shape (A,) ---
        if np.isscalar(initial_price):
            p0 = np.full((self.num_assets,), float(initial_price), dtype=float)
        else:
            p0 = np.asarray(initial_price, dtype=float).reshape(-1)
            if p0.shape[0] != self.num_assets:
                raise ValueError(
                    f"initial_price must be scalar or length {self.num_assets}, got shape {p0.shape}"
                )

        # --- State dim per asset = 1 ---
        initial_state = np.tile(p0[None, :, None], (int(num_trajectories), 1, 1))  # (N,A,1)

        min_value = p0.reshape(self.num_assets, 1)  # (A,1)
        max_value = p0.reshape(self.num_assets, 1)  # (A,1)

        super().__init__(
            min_value=min_value,              # shape (A,1)
            max_value=max_value,              # shape (A,1)
            step_size=float(step_size),
            terminal_time=self.terminal_time,
            initial_state=initial_state,      # shape (N,A,1)
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray = None
    ) -> np.ndarray:
        # Constant -> no change
        return self.current_state

class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: Union[float, np.ndarray] = 0.0, 
        volatility: Union[float, np.ndarray] = 2.0,
        initial_price: Union[float, np.ndarray] = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = num_assets
        self.drift = np.full(num_assets, drift) if np.isscalar(drift) else np.array(drift)
        self.volatility = np.full(num_assets, volatility) if np.isscalar(volatility) else np.array(volatility)
        self.terminal_time = terminal_time

        initial_price = np.full(num_assets, initial_price) if np.isscalar(initial_price) else np.array(initial_price)

        initial_state = np.tile(initial_price[None, :, None], (num_trajectories, 1, 1))  # (num_trajectories, num_assets, 1)
        
        min_value = (initial_price - 4 * self.volatility * np.sqrt(terminal_time)).reshape(num_assets, 1)  # (num_assets, 1)
        max_value = (initial_price + 4 * self.volatility * np.sqrt(terminal_time)).reshape(num_assets, 1)  # (num_assets, 1)
        
        super().__init__(
            min_value=min_value,                  # shape (num_assets, state_dim)
            max_value=max_value,                  # shape (num_assets, state_dim)
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=initial_state,          # shape (num_trajectories, num_assets, state_dim)
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        dt = self.step_size
        # Generate standard normal noise: shape (num_trajectories, num_assets, 1)
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets, 1))
    
        # Drift and diffusion (same logic as single-asset, now extended over assets)
        self.current_state = (
            self.current_state
            + self.drift[None, :, None] * dt
            + self.volatility[None, :, None] * np.sqrt(dt) * noise
        )


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    """
    Multi-asset Geometric Brownian Motion (GBM) midprice model.

    Continuous-time:
        dS_t = mu * S_t dt + sigma * S_t dW_t

    Discrete Euler step (same logic as original single-asset code):
        S_{t+dt} = S_t + mu * S_t * dt + sigma * S_t * sqrt(dt) * Z

    Shapes:
      - current_state: (num_trajectories, num_assets, 1)
      - min_value/max_value: (num_assets, 1)
    """

    def __init__(
        self,
        drift: Union[float, np.ndarray] = 0.0,
        volatility: Union[float, np.ndarray] = 0.1,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.terminal_time = float(terminal_time)

        # --- drift: scalar or (A,) ---
        if np.isscalar(drift):
            self.drift = np.full((self.num_assets,), float(drift), dtype=float)
        else:
            self.drift = np.asarray(drift, dtype=float).reshape(-1)
            if self.drift.shape[0] != self.num_assets:
                raise ValueError(f"drift must be scalar or length {self.num_assets}, got {self.drift.shape}")

        # --- volatility: scalar or (A,) ---
        if np.isscalar(volatility):
            self.volatility = np.full((self.num_assets,), float(volatility), dtype=float)
        else:
            self.volatility = np.asarray(volatility, dtype=float).reshape(-1)
            if self.volatility.shape[0] != self.num_assets:
                raise ValueError(f"volatility must be scalar or length {self.num_assets}, got {self.volatility.shape}")

        if np.any(self.volatility < 0):
            raise ValueError(f"volatility must be >= 0, got {self.volatility}")

        # --- initial_price: scalar or (A,) ---
        if np.isscalar(initial_price):
            p0 = np.full((self.num_assets,), float(initial_price), dtype=float)
        else:
            p0 = np.asarray(initial_price, dtype=float).reshape(-1)
            if p0.shape[0] != self.num_assets:
                raise ValueError(
                    f"initial_price must be scalar or length {self.num_assets}, got shape {p0.shape}"
                )

        # --- initial_state: (N,A,1) ---
        initial_state = np.tile(p0[None, :, None], (int(num_trajectories), 1, 1))

        # --- bounds: match original intent (mean +/- 4 stdev) per asset ---
        max_value_1d = self._get_max_value(p0, self.terminal_time)                    # (A,)
        min_value_1d = p0 - (max_value_1d - p0)                                       # (A,)

        min_value = min_value_1d.reshape(self.num_assets, 1)                          # (A,1)
        max_value = max_value_1d.reshape(self.num_assets, 1)                          # (A,1)

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=float(step_size),
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
        actions: np.ndarray,
        state: np.ndarray = None
    ) -> np.ndarray:
        dt = float(self.step_size)
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets, 1))

        self.current_state = (
            self.current_state
            + self.drift[None, :, None] * self.current_state * dt
            + self.volatility[None, :, None] * self.current_state * sqrt(dt) * noise
        )
        return self.current_state

    def _get_max_value(self, initial_price: np.ndarray, terminal_time: float) -> np.ndarray:
        """
        Vectorized multi-asset version of the original bound heuristic:

        stdev = sqrt(S0^2 * exp(2*mu*T) * (exp(sigma^2*T) - 1))
        max   = S0 * exp(mu*T) + 4*stdev
        """
        S0 = np.asarray(initial_price, dtype=float).reshape(-1)   # (A,)
        T = float(terminal_time)
        mu = self.drift.reshape(-1)                                # (A,)
        sig = self.volatility.reshape(-1)                          # (A,)

        stdev = np.sqrt(
            (S0 ** 2)
            * np.exp(2.0 * mu * T)
            * (np.exp((sig ** 2) * T) - 1.0)
        )
        return (S0 * np.exp(mu * T) + 4.0 * stdev)


class OuMidpriceModel(MidpriceModel):
    """
    Multi-asset Ornstein–Uhlenbeck (OU) midprice model.

    Continuous-time idea (per asset i):
        dP_i = kappa_i (mu_i - P_i) dt + sigma_i dW_i

    Discretized (Euler):
        P_{t+dt} = P_t + kappa*(mu - P_t)*dt + sigma*sqrt(dt)*N(0,1)

    Shapes:
      - current_state: (num_trajectories, num_assets, 1)
      - min_value/max_value: (num_assets, 1)
      - initial_state: (num_trajectories, num_assets, 1)
    """

    def __init__(
        self,
        mean_reversion_level: Union[float, np.ndarray] = 0.0,
        mean_reversion_speed: Union[float, np.ndarray] = 1.0,
        volatility: Union[float, np.ndarray] = 2.0,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.terminal_time = float(terminal_time)

        # --- Expand params to shape (A,) ---
        def _as_A(x: Union[float, np.ndarray], name: str) -> np.ndarray:
            if np.isscalar(x):
                return np.full((self.num_assets,), float(x), dtype=float)
            arr = np.asarray(x, dtype=float).reshape(-1)
            if arr.shape[0] != self.num_assets:
                raise ValueError(f"{name} must be scalar or length {self.num_assets}, got shape {arr.shape}")
            return arr

        self.mean_reversion_level = _as_A(mean_reversion_level, "mean_reversion_level")     # mu (A,)
        self.mean_reversion_speed = _as_A(mean_reversion_speed, "mean_reversion_speed")     # kappa (A,)
        self.volatility = _as_A(volatility, "volatility")                                   # sigma (A,)
        p0 = _as_A(initial_price, "initial_price")                                           # (A,)

        # --- initial_state: (N,A,1) ---
        initial_state = np.tile(p0[None, :, None], (int(num_trajectories), 1, 1))

        # --- bounds per asset: mimic original logic exactly ---
        # original:
        #   max = initial_price + 4 * vol * terminal_time
        #   min = initial_price - (max - initial_price) = initial_price - 4 * vol * terminal_time
        max_value = self._get_max_value(p0, self.terminal_time).reshape(self.num_assets, 1)
        min_value = (p0 - (max_value.reshape(-1) - p0)).reshape(self.num_assets, 1)

        super().__init__(
            min_value=min_value,                  # (A,1)
            max_value=max_value,                  # (A,1)
            step_size=float(step_size),
            terminal_time=self.terminal_time,
            initial_state=initial_state,          # (N,A,1)
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> np.ndarray:
        dt = float(self.step_size)

        # noise: (N,A,1)
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets, 1))

        # broadcast (A,) -> (1,A,1)
        kappa = self.mean_reversion_speed[None, :, None]
        mu = self.mean_reversion_level[None, :, None]
        sigma = self.volatility[None, :, None]

        # Keep the SAME logic as your single-asset code, now vectorized over assets.
        self.current_state += (
            -kappa * (self.current_state - mu) * dt
            + sigma * sqrt(dt) * noise
        )

        return self.current_state

    def _get_max_value(self, initial_price: np.ndarray, terminal_time: float) -> np.ndarray:
        """
        Keep the exact same heuristic as the original single-asset version, applied per asset:
            max = initial_price + 4 * volatility * terminal_time
        """
        return initial_price + 4.0 * self.volatility * float(terminal_time)



class ShortTermOuAlphaMidpriceModel(MidpriceModel):
    """
    Multi-asset version of ShortTermOuAlphaMidpriceModel.

    Per asset i:
        S_{t+dt}^i = S_t^i + alpha_t^i * dt + sigma_i * sqrt(dt) * eps
        alpha^i follows an OU process (handled by ou_process)

    State per asset (state_dim = 2):
        [midprice, alpha]

    Shapes:
      - current_state: (num_trajectories, num_assets, 2)
      - min_value/max_value: (num_assets, 2)
      - initial_state: (num_trajectories, num_assets, 2)
    """

    def __init__(
        self,
        volatility: Union[float, np.ndarray] = 2.0,
        ou_process: Optional[MidpriceModel] = None,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.num_trajectories = int(num_trajectories)
        self.step_size = float(step_size)
        self.terminal_time = float(terminal_time)

        # ---- volatility -> (A,) ----
        if np.isscalar(volatility):
            self.volatility = np.full((self.num_assets,), float(volatility), dtype=float)
        else:
            self.volatility = np.asarray(volatility, dtype=float).reshape(-1)
            if self.volatility.shape[0] != self.num_assets:
                raise ValueError(
                    f"volatility must be scalar or length {self.num_assets}, got {self.volatility.shape}"
                )

        # ---- initial_price -> (A,) ----
        if np.isscalar(initial_price):
            p0 = np.full((self.num_assets,), float(initial_price), dtype=float)
        else:
            p0 = np.asarray(initial_price, dtype=float).reshape(-1)
            if p0.shape[0] != self.num_assets:
                raise ValueError(
                    f"initial_price must be scalar or length {self.num_assets}, got {p0.shape}"
                )

        # ---- OU alpha process ----
        # We expect a multi-asset OU process with state_dim=1 per asset:
        # current_state shape (N, A, 1)
        #
        # If not provided, we create a default OuMidpriceModel with initial_price=0 for each asset.
        # IMPORTANT: we pass a different seed to the OU process to avoid identical RNG streams.
        if ou_process is None:
            # Import here to avoid circular imports if you place this in the same file.
            from mbt_gym.stochastic_processes.N_midprice_models import OuMidpriceModel  # adjust if needed

            ou_process = OuMidpriceModel(
                mean_reversion_level=0.0,
                mean_reversion_speed=1.0,
                volatility=1.0,
                initial_price=np.zeros(self.num_assets, dtype=float),
                terminal_time=terminal_time,
                step_size=step_size,
                num_trajectories=num_trajectories,
                num_assets=num_assets,
                seed=None if seed is None else int(seed) + 12345,
            )

        self.ou_process = ou_process

        # ---- sanity checks on ou_process shapes ----
        # We only need alpha values: (N,A,1)
        if getattr(self.ou_process, "current_state", None) is None:
            raise ValueError("ou_process must be an initialized StochasticProcessModel with current_state.")
        if self.ou_process.current_state.shape != (self.num_trajectories, self.num_assets, 1):
            raise ValueError(
                f"ou_process.current_state must be shape ({self.num_trajectories},{self.num_assets},1), "
                f"got {self.ou_process.current_state.shape}"
            )

        # ---- Build initial_state: (N,A,2) = [price, alpha] ----
        alpha0 = self.ou_process.initial_state[:, :, 0]  # (N,A)
        initial_state = np.zeros((self.num_trajectories, self.num_assets, 2), dtype=float)
        initial_state[:, :, 0] = p0[None, :]
        initial_state[:, :, 1] = alpha0

        # ---- min/max per asset for both components ----
        # Price bounds: same logic as original (rough heuristic)
        p_max = self._get_max_asset_price(p0, terminal_time)          # (A,)
        p_min = p0 - (p_max - p0)                                     # (A,)

        # Alpha bounds come from OU process min/max (shape (A,1))
        # We store alpha in our second component so we need (A,)
        alpha_min = np.asarray(self.ou_process.min_value).reshape(self.num_assets, -1)[:, 0]
        alpha_max = np.asarray(self.ou_process.max_value).reshape(self.num_assets, -1)[:, 0]

        min_value = np.zeros((self.num_assets, 2), dtype=float)
        max_value = np.zeros((self.num_assets, 2), dtype=float)
        min_value[:, 0] = p_min
        max_value[:, 0] = p_max
        min_value[:, 1] = alpha_min
        max_value[:, 1] = alpha_max

        super().__init__(
            min_value=min_value,             # (A,2)
            max_value=max_value,             # (A,2)
            step_size=self.step_size,
            terminal_time=self.terminal_time,
            initial_state=initial_state,     # (N,A,2)
            num_trajectories=self.num_trajectories,
            num_assets=self.num_assets,
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray = None
    ) -> np.ndarray:
        dt = self.step_size

        # alpha_t: (N,A,1) -> (N,A)
        alpha = self.ou_process.current_state[:, :, 0]

        # noise for price: (N,A)
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets))

        # Update price component
        self.current_state[:, :, 0] = (
            self.current_state[:, :, 0]
            + alpha * dt
            + self.volatility[None, :] * sqrt(dt) * noise
        )

        # Update OU alpha process, then copy into our state
        self.ou_process.update(arrivals, fills, actions)
        self.current_state[:, :, 1] = self.ou_process.current_state[:, :, 0]

        return self.current_state

    def _get_max_asset_price(self, initial_price: np.ndarray, terminal_time: float) -> np.ndarray:
        # Vectorized version of the original heuristic
        # returns shape (A,)
        return np.asarray(initial_price, dtype=float) + 4.0 * self.volatility * float(terminal_time)



class BrownianMotionJumpMidpriceModel(MidpriceModel):
    """
    Multi-asset Brownian motion midprice with endogenous jumps driven by (arrivals * fills).

    Per asset i:
        S_{t+dt}^i = S_t^i
                    + drift_i * dt
                    + vol_i * sqrt(dt) * Z
                    + jump_size_i * (fills_ask_i - fills_bid_i)

    Shapes (N-style):
      - current_state: (num_trajectories, num_assets, 1)
      - arrivals:      (num_trajectories, num_assets, 2)
      - fills:         (num_trajectories, num_assets, 2)
    """

    def __init__(
        self,
        drift: Union[float, np.ndarray] = 0.0,
        volatility: Union[float, np.ndarray] = 2.0,
        jump_size: Union[float, np.ndarray] = 1.0,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        if self.num_assets < 1:
            raise ValueError(f"num_assets must be >= 1, got {num_assets}")

        self.terminal_time = float(terminal_time)
        self.step_size = float(step_size)

        # ---- expand params to (A,) ----
        def _as_asset_vec(x, name: str) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.ndim == 0:
                return np.full((self.num_assets,), float(x), dtype=float)
            x = x.reshape(-1)
            if x.shape[0] != self.num_assets:
                raise ValueError(f"{name} must be scalar or length {self.num_assets}, got shape {x.shape}")
            return x

        self.drift = _as_asset_vec(drift, "drift")
        self.volatility = _as_asset_vec(volatility, "volatility")
        self.jump_size = _as_asset_vec(jump_size, "jump_size")
        p0 = _as_asset_vec(initial_price, "initial_price")

        # state_dim per asset = 1 (midprice)
        initial_state = np.tile(p0[None, :, None], (int(num_trajectories), 1, 1))  # (N,A,1)

        # bounds (simple heuristic consistent with the single-asset code style)
        max_value = self._get_max_value(p0, self.terminal_time).reshape(self.num_assets, 1)  # (A,1)
        min_value = (p0 - (max_value.reshape(-1) - p0)).reshape(self.num_assets, 1)          # symmetric around p0

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
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> np.ndarray:
        # Expect (N,A,2)
        if arrivals.shape != (self.num_trajectories, self.num_assets, 2):
            raise ValueError(
                f"Expected arrivals shape ({self.num_trajectories},{self.num_assets},2), got {arrivals.shape}"
            )
        if fills.shape != (self.num_trajectories, self.num_assets, 2):
            raise ValueError(
                f"Expected fills shape ({self.num_trajectories},{self.num_assets},2), got {fills.shape}"
            )

        dt = self.step_size

        # Brownian term
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets, 1))

        # IMPORTANT: make numeric before subtraction (fills/arrivals can be bool)
        arrivals_f = arrivals.astype(float)
        fills_f = fills.astype(float)

        # Endogenous jump term: executed fills only
        fills_bid = fills_f[:, :, BID_INDEX] * arrivals_f[:, :, BID_INDEX]  # (N,A)
        fills_ask = fills_f[:, :, ASK_INDEX] * arrivals_f[:, :, ASK_INDEX]  # (N,A)
        jump_term = (self.jump_size[None, :] * (fills_ask - fills_bid))[:, :, None]  # (N,A,1)

        self.current_state = (
            self.current_state
            + self.drift[None, :, None] * dt
            + self.volatility[None, :, None] * sqrt(dt) * noise
            + jump_term
        )

        # Optional safety clip (matches your other N-style processes)
        self.current_state = np.clip(
            self.current_state,
            self.min_value[None, :, :],
            self.max_value[None, :, :],
        )
        return self.current_state

    def _get_max_value(self, initial_price_vec: np.ndarray, terminal_time: float) -> np.ndarray:
        # same heuristic as single-asset: p0 + 4*sigma*T
        return initial_price_vec + 4.0 * self.volatility * float(terminal_time)


class OuJumpMidpriceModel(MidpriceModel):
    """
    Multi-asset OU midprice model with endogenous jumps driven by executed order flow (arrivals * fills).

    Per asset i:
        S_{t+dt}^i = S_t^i
                    - kappa_i * (S_t^i - mu_i)          [NOTE: matches the single-asset code (no dt multiplier)]
                    + sigma_i * sqrt(dt) * Z
                    + jump_i * (fills_ask_i - fills_bid_i)

    Shapes (N-style):
      - current_state: (num_trajectories, num_assets, 1)
      - arrivals:      (num_trajectories, num_assets, 2)
      - fills:         (num_trajectories, num_assets, 2)

    This version introduces NO cross-asset coupling.
    """

    def __init__(
        self,
        mean_reversion_level: Union[float, np.ndarray] = 0.0,
        mean_reversion_speed: Union[float, np.ndarray] = 1.0,
        volatility: Union[float, np.ndarray] = 2.0,
        jump_size: Union[float, np.ndarray] = 1.0,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.terminal_time = float(terminal_time)
        self.step_size = float(step_size)

        def _as_asset_vec(x, name: str) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.ndim == 0:
                return np.full((self.num_assets,), float(x), dtype=float)
            x = x.reshape(-1)
            if x.shape[0] != self.num_assets:
                raise ValueError(f"{name} must be scalar or length {self.num_assets}, got shape {x.shape}")
            return x

        self.mean_reversion_level = _as_asset_vec(mean_reversion_level, "mean_reversion_level")   # mu_i
        self.mean_reversion_speed = _as_asset_vec(mean_reversion_speed, "mean_reversion_speed")   # kappa_i
        self.volatility = _as_asset_vec(volatility, "volatility")                                  # sigma_i
        self.jump_size = _as_asset_vec(jump_size, "jump_size")                                      # J_i

        p0 = _as_asset_vec(initial_price, "initial_price")

        # State dim per asset = 1 (midprice)
        initial_state = np.tile(p0[None, :, None], (int(num_trajectories), 1, 1))  # (N,A,1)

        # Bounds: follow the single-asset heuristic p0 +/- (max - p0), where max = p0 + 4*sigma*T
        max_value = self._get_max_value(p0, self.terminal_time).reshape(self.num_assets, 1)  # (A,1)
        min_value = (p0 - (max_value.reshape(-1) - p0)).reshape(self.num_assets, 1)          # (A,1)

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
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> np.ndarray:
        # Expect (N,A,2)
        if arrivals.shape != (self.num_trajectories, self.num_assets, 2):
            raise ValueError(
                f"Expected arrivals shape ({self.num_trajectories},{self.num_assets},2), got {arrivals.shape}"
            )
        if fills.shape != (self.num_trajectories, self.num_assets, 2):
            raise ValueError(
                f"Expected fills shape ({self.num_trajectories},{self.num_assets},2), got {fills.shape}"
            )

        dt = self.step_size

        # Diffusion noise: (N,A,1)
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets, 1))

        # Executed fills only (convert bool -> float to allow subtraction)
        fills_bid = fills[:, :, BID_INDEX].astype(float) * arrivals[:, :, BID_INDEX].astype(float)  # (N,A)
        fills_ask = fills[:, :, ASK_INDEX].astype(float) * arrivals[:, :, ASK_INDEX].astype(float)  # (N,A)

        jump_term = (self.jump_size[None, :] * (fills_ask - fills_bid))[:, :, None]  # (N,A,1)

        # Mean reversion term (matches your single-asset code: NO *dt here)
        mr_term = -self.mean_reversion_speed[None, :, None] * (
            self.current_state - self.mean_reversion_level[None, :, None]
        )

        # Update
        self.current_state = (
            self.current_state
            + mr_term
            + self.volatility[None, :, None] * sqrt(dt) * noise
            + jump_term
        )

        # Clip to bounds
        self.current_state = np.clip(
            self.current_state,
            self.min_value[None, :, :],
            self.max_value[None, :, :],
        )
        return self.current_state

    def _get_max_value(self, initial_price_vec: np.ndarray, terminal_time: float) -> np.ndarray:
        # same heuristic as single-asset: p0 + 4*sigma*T
        return initial_price_vec + 4.0 * self.volatility * float(terminal_time)

class ShortTermJumpAlphaMidpriceModel(MidpriceModel):
    """
    Multi-asset version of ShortTermJumpAlphaMidpriceModel.

    State per asset has 2 dims:
      - state[..., 0] = midprice S_t
      - state[..., 1] = alpha_t (from ou_jump_process)

    Shapes (N-style):
      - current_state:     (num_trajectories, num_assets, 2)
      - ou_jump_process.current_state: (num_trajectories, num_assets, 1)
      - arrivals/fills:    (num_trajectories, num_assets, 2)
    """

    def __init__(
        self,
        volatility: Union[float, np.ndarray] = 2.0,
        ou_jump_process: "OuJumpMidpriceModel" = None,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.terminal_time = float(terminal_time)
        self.step_size = float(step_size)

        def _as_asset_vec(x, name: str) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.ndim == 0:
                return np.full((self.num_assets,), float(x), dtype=float)
            x = x.reshape(-1)
            if x.shape[0] != self.num_assets:
                raise ValueError(f"{name} must be scalar or length {self.num_assets}, got shape {x.shape}")
            return x

        self.volatility = _as_asset_vec(volatility, "volatility")
        p0 = _as_asset_vec(initial_price, "initial_price")

        # Use provided OU-jump alpha process or build a default one.
        # IMPORTANT: default assumes your multi-asset OuJumpMidpriceModel exists and accepts num_assets.
        self.ou_jump_process = ou_jump_process or OuJumpMidpriceModel(
            mean_reversion_level=np.zeros(self.num_assets),
            mean_reversion_speed=np.ones(self.num_assets),
            volatility=np.ones(self.num_assets),
            jump_size=np.ones(self.num_assets),
            initial_price=np.zeros(self.num_assets),   # alpha starts at 0
            terminal_time=self.terminal_time,
            step_size=self.step_size,
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

        # --- bounds for price component ---
        price_max = self._get_max_asset_price(p0, self.terminal_time).reshape(self.num_assets, 1)  # (A,1)
        price_min = (p0 - (price_max.reshape(-1) - p0)).reshape(self.num_assets, 1)               # (A,1)

        # --- bounds for alpha component from OU-jump process ---
        alpha_min = np.asarray(self.ou_jump_process.min_value, dtype=float).reshape(self.num_assets, 1)  # (A,1)
        alpha_max = np.asarray(self.ou_jump_process.max_value, dtype=float).reshape(self.num_assets, 1)  # (A,1)

        min_value = np.concatenate([price_min, alpha_min], axis=1)  # (A,2)
        max_value = np.concatenate([price_max, alpha_max], axis=1)  # (A,2)

        # initial alpha (A,) from the alpha process initial_state (N,A,1) -> take first traj
        alpha0 = np.asarray(self.ou_jump_process.initial_state[0, :, 0], dtype=float)  # (A,)

        # initial_state: (N,A,2)
        initial_state_one = np.stack([p0, alpha0], axis=1)  # (A,2)
        initial_state = np.tile(initial_state_one[None, :, :], (int(num_trajectories), 1, 1))  # (N,A,2)

        super().__init__(
            min_value=min_value,                 # (A,2)
            max_value=max_value,                 # (A,2)
            step_size=self.step_size,
            terminal_time=self.terminal_time,
            initial_state=initial_state,         # (N,A,2)
            num_trajectories=int(num_trajectories),
            num_assets=self.num_assets,
            seed=seed,
        )

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> np.ndarray:
        dt = self.step_size

        # alpha from OU-jump process: (N,A,1)
        alpha = self.ou_jump_process.current_state

        # Brownian noise for price: (N,A,1)
        noise = self.rng.normal(size=(self.num_trajectories, self.num_assets, 1))

        # Update PRICE component using current alpha
        self.current_state[:, :, 0:1] = (
            self.current_state[:, :, 0:1]
            + alpha * dt
            + self.volatility[None, :, None] * sqrt(dt) * noise
        )

        # Update alpha process (OU + jumps driven by arrivals*fills)
        self.ou_jump_process.update(arrivals, fills, actions, state)

        # Copy alpha into state alpha component
        self.current_state[:, :, 1:2] = self.ou_jump_process.current_state

        # Optional clip (consistent with your other N-style models)
        self.current_state = np.clip(
            self.current_state,
            self.min_value[None, :, :],
            self.max_value[None, :, :],
        )
        return self.current_state

    def _get_max_asset_price(self, initial_price_vec: np.ndarray, terminal_time: float) -> np.ndarray:
        # Same heuristic style as the single-asset version
        return initial_price_vec + 4.0 * self.volatility * float(terminal_time)


class HestonMidpriceModel(MidpriceModel):
    """
    Multi-asset Heston-style midprice model.

    State per asset:
        [price, variance]

    Shapes:
      - current_state: (N, A, 2)
      - min_value/max_value: (A, 2)

    Notes:
      - Keeps the original logic: per-asset correlated (price, variance) shocks.
      - Assumes assets are independent of each other (no cross-asset correlation).
    """

    def __init__(
        self,
        drift: Union[float, np.ndarray] = 0.05,
        volatility_mean_reversion_rate: Union[float, np.ndarray] = 3.0,
        volatility_mean_reversion_level: Union[float, np.ndarray] = 0.04,
        weiner_correlation: Union[float, np.ndarray] = -0.8,
        volatility_of_volatility: Union[float, np.ndarray] = 0.6,
        initial_price: Union[float, np.ndarray] = 100.0,
        initial_variance: Union[float, np.ndarray] = 0.2**2,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_assets = int(num_assets)
        self.terminal_time = float(terminal_time)

        # --------- helper: expand scalar -> (A,) ----------
        def _expand(x, name: str) -> np.ndarray:
            x = np.array(x, dtype=float)
            if x.shape == ():  # scalar
                return np.full((self.num_assets,), float(x))
            if x.shape == (self.num_assets,):
                return x
            raise ValueError(f"{name} must be scalar or shape ({self.num_assets},), got {x.shape}")

        self.drift = _expand(drift, "drift")
        self.volatility_mean_reversion_rate = _expand(volatility_mean_reversion_rate, "volatility_mean_reversion_rate")
        self.volatility_mean_reversion_level = _expand(volatility_mean_reversion_level, "volatility_mean_reversion_level")
        self.weiner_correlation = _expand(weiner_correlation, "weiner_correlation")
        self.volatility_of_volatility = _expand(volatility_of_volatility, "volatility_of_volatility")

        init_price = _expand(initial_price, "initial_price")
        init_var = _expand(initial_variance, "initial_variance")

        # --------- min/max bounds (same heuristic as your single-asset code) ----------
        max_price = self._get_max_value(init_price, self.terminal_time)  # (A,)
        min_price = init_price - (max_price - init_price)                # symmetric around initial

        # Provide bounds for both dimensions: [price, variance]
        # Variance bounds are heuristic; keep simple but safe.
        min_value = np.stack([min_price, np.zeros_like(init_var)], axis=1)  # (A,2)
        max_value = np.stack([max_price, np.full_like(init_var, np.max(init_var) * 10.0 + 1e-6)], axis=1)  # (A,2)

        # --------- initial state: (N, A, 2) ----------
        initial_state = np.zeros((num_trajectories, self.num_assets, 2), dtype=float)
        initial_state[:, :, 0] = init_price[None, :]
        initial_state[:, :, 1] = init_var[None, :]

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            step_size=float(step_size),
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
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> None:
        """
        Vectorized per-asset update with per-asset (price, var) correlation.
        """
        dt = self.step_size

        S = self.current_state[:, :, 0]  # (N,A)
        v = self.current_state[:, :, 1]  # (N,A)

        # Draw correlated normals for each (trajectory, asset)
        # eps1 ~ N(0,1), eps2 = rho*eps1 + sqrt(1-rho^2)*z
        eps1 = self.rng.standard_normal(size=S.shape)  # (N,A)
        z = self.rng.standard_normal(size=S.shape)     # (N,A)

        rho = self.weiner_correlation[None, :]  # (1,A)
        rho = np.clip(rho, -0.999999, 0.999999)  # safety

        eps2 = rho * eps1 + np.sqrt(1.0 - rho**2) * z  # (N,A)

        # Price update (same logic)
        S_new = (
            S
            + self.drift[None, :] * S * dt
            + np.sqrt(np.maximum(v, 0.0) * dt) * S * eps1
        )

        # Variance update (same logic; uses abs in original code)
        v_new = np.abs(
            v
            + self.volatility_mean_reversion_rate[None, :]
            * (self.volatility_mean_reversion_level[None, :] - v)
            * dt
            + self.volatility_of_volatility[None, :]
            * np.sqrt(np.maximum(v, 0.0) * dt)
            * eps2
        )

        self.current_state[:, :, 0] = S_new
        self.current_state[:, :, 1] = v_new

        # Optional: clip to bounds to match your other models' style
        self.current_state = np.clip(self.current_state, self.min_value[None, :, :], self.max_value[None, :, :])

    def _get_max_value(self, initial_price: np.ndarray, terminal_time: float) -> np.ndarray:
        # Same heuristic as your original method, now vectorized
        return initial_price + 4.0 * self.volatility_mean_reversion_level * terminal_time


class ConstantElasticityOfVarianceMidpriceModel(MidpriceModel):
    """
    Multi-asset CEV model (Euler-Maruyama).

    Continuous-time:
        dS = mu * S dt + sigma * S^gamma dW

    Discrete:
        S_{t+dt} = S_t + mu*S_t*dt + sigma*(S_t^gamma)*sqrt(dt)*Z
    """

    def __init__(
        self,
        drift: Union[float, np.ndarray] = 0.0,
        volatility: Union[float, np.ndarray] = 0.1,
        gamma: Union[float, np.ndarray] = 1.0,
        initial_price: Union[float, np.ndarray] = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        num_assets: int = 1,
        seed: Optional[int] = None,
    ):
        self.terminal_time = terminal_time
        self.num_assets = num_assets

        # --- broadcast params to shape (A,) ---
        self.drift = self._as_asset_vector(drift, num_assets)
        self.volatility = self._as_asset_vector(volatility, num_assets)
        self.gamma = self._as_asset_vector(gamma, num_assets)

        init_price = self._as_asset_vector(initial_price, num_assets)

        # Bounds per asset (shape (A,1)) to match your env usage: max_value[:,0]
        max_v = init_price + 4.0 * self.volatility * terminal_time
        min_v = init_price - (max_v - init_price)

        # initial_state shape (N, A, 1)
        initial_state = np.tile(init_price.reshape(1, num_assets, 1), (num_trajectories, 1, 1))

        super().__init__(
            min_value=min_v.reshape(num_assets, 1),
            max_value=max_v.reshape(num_assets, 1),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            seed=seed,
        )

    @staticmethod
    def _as_asset_vector(x: Union[float, np.ndarray], num_assets: int) -> np.ndarray:
        """Convert scalar or array-like to shape (A,) float array."""
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.full((num_assets,), float(arr))
        if arr.shape == (num_assets,):
            return arr
        raise ValueError(f"Expected scalar or shape ({num_assets},), got {arr.shape}")

    def update(
        self,
        arrivals: np.ndarray,
        fills: np.ndarray,
        actions: np.ndarray,
        state: np.ndarray = None,
    ) -> np.ndarray:
        """
        Updates self.current_state in-place.

        current_state shape: (N, A, 1)
        """
        S = self.current_state[:, :, 0]  # (N, A)

        # i.i.d. shocks per trajectory and asset
        Z = self.rng.normal(size=(self.num_trajectories, self.num_assets))  # (N, A)

        dt = self.step_size
        mu = self.drift.reshape(1, self.num_assets)        # (1, A)
        sig = self.volatility.reshape(1, self.num_assets)  # (1, A)
        gam = self.gamma.reshape(1, self.num_assets)       # (1, A)

        S_next = (
            S
            + mu * S * dt
            + sig * (S ** gam) * np.sqrt(dt) * Z
        )

        self.current_state[:, :, 0] = S_next
        return self.current_state
