from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from mbt_gym.gym.N_Bek_TradingEnvironment import Bek_TradingEnvironment
from mbt_gym.rewards.N_RewardFunctions import CjMmCriterion
from mbt_gym.stochastic_processes.N_midprice_models import BrownianMotionMidpriceModel
from mbt_gym.stochastic_processes.N_arrival_models import (
    SynchronousHawkesArrivalModel,
    SynchronousLOBDepthModel,
)
from mbt_gym.stochastic_processes.N_fill_probability_models import DynamicLOBExponentialFillFunction
from mbt_gym.gym.N_Bek_ModelDynamics import Bek_LimitOrderModelDynamics


# ============================================================
# Global config container (so builders don't depend on notebook)
# ============================================================
@dataclass
class EnvGlobals:
    terminal_time: float = 1.0
    n_steps: int = 100
    phi: float = 0.005
    alpha: float = 0.2
    train_seed: int = 123


_CFG = EnvGlobals()


def set_env_globals(
    *,
    terminal_time: Optional[float] = None,
    n_steps: Optional[int] = None,
    phi: Optional[float] = None,
    alpha: Optional[float] = None,
    train_seed: Optional[int] = None,
) -> None:
    """
    Call this once in your notebook before building envs, e.g.

    set_env_globals(
        terminal_time=terminal_time,
        n_steps=n_steps,
        phi=phi,
        alpha=alpha,
        train_seed=train_seed,
    )
    """
    global _CFG
    if terminal_time is not None:
        _CFG.terminal_time = float(terminal_time)
    if n_steps is not None:
        _CFG.n_steps = int(n_steps)
    if phi is not None:
        _CFG.phi = float(phi)
    if alpha is not None:
        _CFG.alpha = float(alpha)
    if train_seed is not None:
        _CFG.train_seed = int(train_seed)


# ============================================================
# N=1
# ============================================================
def get_cj_env_N1(
    *,
    num_trajectories: int = 1,
    num_assets: int = 1,
    initial_inventory=None,
    seed: Optional[int] = None,
    sigma=None,
    initial_price=None,
    baseline_arrival_rate=None,
    cross_asset_influence=None,
    mean_reversion_speed=None,
    self_jump_size=None,
    mutual_jump_size=None,
    synchrony_factor=None,
    c_baseline_depth=None,
    c_cross_asset_influence=None,
    c_mean_reversion_speed=None,
    c_self_jump_size=None,
    c_mutual_jump_size=None,
    c_synchrony_factor=None,
) -> Bek_TradingEnvironment:
    """
    Single-asset (N=1) environment builder.
    - Uses globals from _CFG (set via set_env_globals)
    - If seed is provided, it overrides _CFG.train_seed
    - Arrival and LOB parameters can be overridden from training script
    """
    assert num_assets == 1, "get_cj_env_N1 is intended for num_assets=1."

    terminal_time = _CFG.terminal_time
    n_steps = _CFG.n_steps
    phi = _CFG.phi
    alpha = _CFG.alpha
    train_seed = _CFG.train_seed if seed is None else int(seed)

    # --- Asset-specific params ---
    if sigma is None:
        sigma = 0.1
    if initial_price is None:
        initial_price = 100.0

    if initial_inventory is None:
        initial_inventory = (-3, 4)

    step_size = 1.0 / n_steps

    # --- Midprice model ---
    midprice_model = BrownianMotionMidpriceModel(
        volatility=sigma,
        initial_price=initial_price,
        terminal_time=terminal_time,
        step_size=step_size,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Arrival model defaults ---
    if baseline_arrival_rate is None:
        baseline_arrival_rate = [12.0, 12.0]  # (2,) -> tiled to (1,2)

    if mean_reversion_speed is None:
        mean_reversion_speed = 15.0

    if self_jump_size is None:
        self_jump_size = 3.0

    if mutual_jump_size is None:
        mutual_jump_size = 1.5

    if synchrony_factor is None:
        synchrony_factor = 0.05

    if cross_asset_influence is None:
        cross_asset_influence = 0.0  # scalar -> expands to (1,1)

    arrival_model = SynchronousHawkesArrivalModel(
        step_size=step_size,
        terminal_time=terminal_time,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        baseline_arrival_rate=baseline_arrival_rate,
        mean_reversion_speed=mean_reversion_speed,
        self_jump_size=self_jump_size,
        mutual_jump_size=mutual_jump_size,
        synchrony_factor=synchrony_factor,
        cross_asset_influence=cross_asset_influence,
    )

    # --- LOB depth model defaults ---
    if c_baseline_depth is None:
        c_baseline_depth = [2.0, 2.0]  # (2,) -> tiled to (1,2)

    if c_mean_reversion_speed is None:
        c_mean_reversion_speed = 10.0

    if c_self_jump_size is None:
        c_self_jump_size = 0.6

    if c_mutual_jump_size is None:
        c_mutual_jump_size = 0.3

    if c_synchrony_factor is None:
        c_synchrony_factor = 0.05

    if c_cross_asset_influence is None:
        c_cross_asset_influence = 0.0  # scalar -> expands to (1,1)

    lob_depth_model = SynchronousLOBDepthModel(
        step_size=step_size,
        terminal_time=terminal_time,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        c_baseline_depth=c_baseline_depth,
        c_mean_reversion_speed=c_mean_reversion_speed,
        c_self_jump_size=c_self_jump_size,
        c_mutual_jump_size=c_mutual_jump_size,
        c_synchrony_factor=c_synchrony_factor,
        c_cross_asset_influence=c_cross_asset_influence,
    )

    # --- Fill probability model ---
    fill_probability_model = DynamicLOBExponentialFillFunction(
        lob_depth_model=lob_depth_model,
        step_size=step_size,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Model dynamics ---
    model_dynamics = Bek_LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        fill_probability_model=fill_probability_model,
        lob_depth_model=lob_depth_model,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Reward ---
    reward_function = CjMmCriterion(
        per_step_inventory_aversion=phi,
        terminal_inventory_aversion=alpha,
        num_assets=num_assets,
    )

    env_params = dict(
        terminal_time=terminal_time,
        n_steps=n_steps,
        initial_inventory=initial_inventory,
        model_dynamics=model_dynamics,
        max_inventory=n_steps,
        normalise_action_space=False,
        normalise_observation_space=False,
        normalise_rewards=False,
        reward_function=reward_function,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        seed=train_seed,
    )

    return Bek_TradingEnvironment(**env_params)


# ============================================================
# N=2
# ============================================================
def get_cj_env_N2(
    *,
    num_trajectories: int = 1,
    num_assets: int = 2,
    initial_inventory=None,
    seed: Optional[int] = None,
    sigma=None,
    initial_price=None,
    baseline_arrival_rate=None,
    cross_asset_influence=None,
    mean_reversion_speed=None,
    self_jump_size=None,
    mutual_jump_size=None,
    synchrony_factor=None,
    c_baseline_depth=None,
    c_cross_asset_influence=None,
    c_mean_reversion_speed=None,
    c_self_jump_size=None,
    c_mutual_jump_size=None,
    c_synchrony_factor=None,
) -> Bek_TradingEnvironment:
    """
    Multi-asset (N=2) environment builder.
    - Uses globals from _CFG (set via set_env_globals)
    - If seed is provided, it overrides _CFG.train_seed
    - Arrival and LOB parameters can be overridden from training script
    """
    assert num_assets == 2, "get_cj_env_N2 is intended for num_assets=2."

    terminal_time = _CFG.terminal_time
    n_steps = _CFG.n_steps
    phi = _CFG.phi
    alpha = _CFG.alpha
    train_seed = _CFG.train_seed if seed is None else int(seed)

    # --- params --
    if sigma is None:
        sigma = np.array([0.1, 0.1])
    if initial_price is None:
        initial_price = np.array([100.0, 100.0])

    if initial_inventory is None:
        initial_inventory = (-1, 2)

    step_size = 1.0 / n_steps

    # --- Midprice model ---
    midprice_model = BrownianMotionMidpriceModel(
        volatility=sigma,
        initial_price=initial_price,
        terminal_time=terminal_time,
        step_size=step_size,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Arrival model defaults ---
    if baseline_arrival_rate is None:
        baseline_arrival_rate = [[10.0, 10.0], [12.0, 12.0]]

    if cross_asset_influence is None:
        cross_asset_influence = [[0.0, 0.05], [0.05, 0.0]]

    if mean_reversion_speed is None:
        mean_reversion_speed = np.array([15.0, 18.0])

    if self_jump_size is None:
        self_jump_size = np.array([3.0, 3.2])

    if mutual_jump_size is None:
        mutual_jump_size = np.array([1.5, 1.6])

    if synchrony_factor is None:
        synchrony_factor = np.array([0.05, 0.05])

    arrival_model = SynchronousHawkesArrivalModel(
        step_size=step_size,
        terminal_time=terminal_time,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        baseline_arrival_rate=baseline_arrival_rate,
        mean_reversion_speed=mean_reversion_speed,
        self_jump_size=self_jump_size,
        mutual_jump_size=mutual_jump_size,
        synchrony_factor=synchrony_factor,
        cross_asset_influence=cross_asset_influence,
    )

    # --- LOB depth model defaults ---
    if c_baseline_depth is None:
        c_baseline_depth = [[2.0, 2.0], [2.5, 2.5]]

    if c_cross_asset_influence is None:
        c_cross_asset_influence = [[0.0, 0.05], [0.05, 0.0]]

    if c_mean_reversion_speed is None:
        c_mean_reversion_speed = np.array([10.0, 12.0])

    if c_self_jump_size is None:
        c_self_jump_size = np.array([0.6, 0.65])

    if c_mutual_jump_size is None:
        c_mutual_jump_size = np.array([0.3, 0.32])

    if c_synchrony_factor is None:
        c_synchrony_factor = np.array([0.05, 0.05])

    lob_depth_model = SynchronousLOBDepthModel(
        step_size=step_size,
        terminal_time=terminal_time,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        c_baseline_depth=c_baseline_depth,
        c_mean_reversion_speed=c_mean_reversion_speed,
        c_self_jump_size=c_self_jump_size,
        c_mutual_jump_size=c_mutual_jump_size,
        c_synchrony_factor=c_synchrony_factor,
        c_cross_asset_influence=c_cross_asset_influence,
    )

    # --- Fill probability model ---
    fill_probability_model = DynamicLOBExponentialFillFunction(
        lob_depth_model=lob_depth_model,
        step_size=step_size,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Model dynamics ---
    model_dynamics = Bek_LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        fill_probability_model=fill_probability_model,
        lob_depth_model=lob_depth_model,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Reward ---
    reward_function = CjMmCriterion(
        per_step_inventory_aversion=phi,
        terminal_inventory_aversion=alpha,
        num_assets=num_assets,
    )

    env_params = dict(
        terminal_time=terminal_time,
        n_steps=n_steps,
        initial_inventory=initial_inventory,
        model_dynamics=model_dynamics,
        max_inventory=n_steps,
        normalise_action_space=False,
        normalise_observation_space=False,
        normalise_rewards=False,
        reward_function=reward_function,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        seed=train_seed,
    )

    return Bek_TradingEnvironment(**env_params)


# ============================================================
# N=3
# ============================================================
def get_cj_env_N3(
    *,
    num_trajectories: int = 1,
    num_assets: int = 3,
    initial_inventory=None,
    seed: Optional[int] = None,
    sigma=None,
    initial_price=None,
    baseline_arrival_rate=None,
    cross_asset_influence=None,
    mean_reversion_speed=None,
    self_jump_size=None,
    mutual_jump_size=None,
    synchrony_factor=None,
    c_baseline_depth=None,
    c_cross_asset_influence=None,
    c_mean_reversion_speed=None,
    c_self_jump_size=None,
    c_mutual_jump_size=None,
    c_synchrony_factor=None,
) -> Bek_TradingEnvironment:
    """
    Multi-asset (N=3) environment builder.
    - Uses globals from _CFG (set via set_env_globals)
    - If seed is provided, it overrides _CFG.train_seed
    - Arrival and LOB parameters can be overridden from training script
    """
    assert num_assets == 3, "get_cj_env_N3 is intended for num_assets=3."

    terminal_time = _CFG.terminal_time
    n_steps = _CFG.n_steps
    phi = _CFG.phi
    alpha = _CFG.alpha
    train_seed = _CFG.train_seed if seed is None else int(seed)

    if sigma is None:
        sigma = np.array([0.1, 0.1, 0.1])
    if initial_price is None:
        initial_price = np.array([100.0, 100.0, 100.0])

    if initial_inventory is None:
        initial_inventory = (-1, 2)

    step_size = 1.0 / n_steps

    # --- Midprice model ---
    midprice_model = BrownianMotionMidpriceModel(
        volatility=sigma,
        initial_price=initial_price,
        terminal_time=terminal_time,
        step_size=step_size,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Arrival model defaults ---
    if baseline_arrival_rate is None:
        baseline_arrival_rate = [
            [10.0, 10.0],
            [12.0, 12.0],
            [14.0, 14.0],
        ]

    if cross_asset_influence is None:
        cross_asset_influence = [
            [0.0, 0.05, 0.05],
            [0.05, 0.0, 0.05],
            [0.05, 0.05, 0.0],
        ]

    if mean_reversion_speed is None:
        mean_reversion_speed = np.array([15.0, 18.0, 21.0])

    if self_jump_size is None:
        self_jump_size = np.array([3.0, 3.2, 3.4])

    if mutual_jump_size is None:
        mutual_jump_size = np.array([1.5, 1.6, 1.7])

    if synchrony_factor is None:
        synchrony_factor = np.array([0.05, 0.05, 0.05])

    arrival_model = SynchronousHawkesArrivalModel(
        step_size=step_size,
        terminal_time=terminal_time,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        baseline_arrival_rate=baseline_arrival_rate,
        mean_reversion_speed=mean_reversion_speed,
        self_jump_size=self_jump_size,
        mutual_jump_size=mutual_jump_size,
        synchrony_factor=synchrony_factor,
        cross_asset_influence=cross_asset_influence,
    )

    # --- LOB depth model defaults ---
    if c_baseline_depth is None:
        c_baseline_depth = [
            [2.0, 2.0],
            [2.5, 2.5],
            [3.0, 3.0],
        ]

    if c_cross_asset_influence is None:
        c_cross_asset_influence = [
            [0.0, 0.05, 0.05],
            [0.05, 0.0, 0.05],
            [0.05, 0.05, 0.0],
        ]

    if c_mean_reversion_speed is None:
        c_mean_reversion_speed = np.array([10.0, 12.0, 14.0])

    if c_self_jump_size is None:
        c_self_jump_size = np.array([0.6, 0.65, 0.70])

    if c_mutual_jump_size is None:
        c_mutual_jump_size = np.array([0.3, 0.32, 0.34])

    if c_synchrony_factor is None:
        c_synchrony_factor = np.array([0.05, 0.05, 0.05])

    lob_depth_model = SynchronousLOBDepthModel(
        step_size=step_size,
        terminal_time=terminal_time,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        c_baseline_depth=c_baseline_depth,
        c_mean_reversion_speed=c_mean_reversion_speed,
        c_self_jump_size=c_self_jump_size,
        c_mutual_jump_size=c_mutual_jump_size,
        c_synchrony_factor=c_synchrony_factor,
        c_cross_asset_influence=c_cross_asset_influence,
    )

    # --- Fill probability model ---
    fill_probability_model = DynamicLOBExponentialFillFunction(
        lob_depth_model=lob_depth_model,
        step_size=step_size,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Model dynamics ---
    model_dynamics = Bek_LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        fill_probability_model=fill_probability_model,
        lob_depth_model=lob_depth_model,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
    )

    # --- Reward ---
    reward_function = CjMmCriterion(
        per_step_inventory_aversion=phi,
        terminal_inventory_aversion=alpha,
        num_assets=num_assets,
    )

    env_params = dict(
        terminal_time=terminal_time,
        n_steps=n_steps,
        initial_inventory=initial_inventory,
        model_dynamics=model_dynamics,
        max_inventory=n_steps,
        normalise_action_space=False,
        normalise_observation_space=False,
        normalise_rewards=False,
        reward_function=reward_function,
        num_trajectories=num_trajectories,
        num_assets=num_assets,
        seed=train_seed,
    )

    return Bek_TradingEnvironment(**env_params)


# ============================================================
# Dispatcher: import THIS in your notebook
# ============================================================
def get_cj_env(
    *,
    num_trajectories: int = 1,
    num_assets: int = 1,
    initial_inventory=None,
    seed: Optional[int] = None,
    **kwargs,
) -> Bek_TradingEnvironment:
    """
    Unified entry point.
    In notebook, you always call get_cj_env(num_assets=1/2/3, ...)

    Extra keyword arguments are forwarded to N=1 / N=2 / N=3 builders,
    allowing you to override arrival / LOB parameters directly from
    the training script.
    """
    if num_assets == 1:
        return get_cj_env_N1(
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            initial_inventory=initial_inventory,
            seed=seed,
            **kwargs,
        )
    if num_assets == 2:
        return get_cj_env_N2(
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            initial_inventory=initial_inventory,
            seed=seed,
            **kwargs,
        )
    if num_assets == 3:
        return get_cj_env_N3(
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            initial_inventory=initial_inventory,
            seed=seed,
            **kwargs,
        )

    raise ValueError(f"Unsupported num_assets={num_assets}. Expected 1, 2, or 3.")