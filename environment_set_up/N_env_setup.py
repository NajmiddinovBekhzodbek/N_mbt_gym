# mbt_gym/utils/N_env_setup.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Any, Sequence

import numpy as np

from mbt_gym.gym.N_index_names import get_index_map
from mbt_gym.gym.N_wrappers import ReduceStateSizeWrapper
from mbt_gym.gym.N_StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from mbt_gym.gym.helpers.N_generate_trajectory import generate_trajectory
from mbt_gym.agents.Agent import Agent


# ============================================================
# 1) Robust state selection (works across model combinations)
# ============================================================
def build_state_indices(
    env_raw: Any,
    *,
    num_assets: int,
    include_arrivals: bool = True,
    include_lob: bool = True,
) -> List[int]:
    """
    Always includes inventory (all assets) + time,
    and optionally includes arrival_model + lob_depth_model blocks if present.
    """
    idx = get_index_map(num_assets)

    inventory_indices = list(range(idx["INVENTORY_INDEX"], idx["TIME_INDEX"]))
    time_index = [idx["TIME_INDEX"]]
    state_indices = inventory_indices + time_index

    spi = getattr(env_raw, "stochastic_process_indices", {})

    if include_arrivals and "arrival_model" in spi:
        a0, a1 = spi["arrival_model"]
        state_indices += list(range(a0, a1))

    if include_lob and "lob_depth_model" in spi:
        c0, c1 = spi["lob_depth_model"]
        state_indices += list(range(c0, c1))

    return state_indices


# ============================================================
# 2) Baseline reward estimate (for reward scaling)
# ============================================================
class _FixedDepthAgent(Agent):
    """
    Fixed posting depth agent:
      action shape expected: (M, A, 2) where 2 is (bid, ask)
    """

    def __init__(self, depth: float, action_shape: Tuple[int, int]):
        self.depth = float(depth)
        self.A, self.two = action_shape

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.ones((obs.shape[0], self.A, self.two), dtype=float) * self.depth


def _mean_episode_reward_from_rewards(
    rewards: np.ndarray,
    *,
    num_total_trajectories: int,
    n_steps: Optional[int] = None,
) -> float:
    """
    Convert rewards array into mean episode reward:
        baseline_mean = E[ sum_t sum_assets r_t ] averaged over trajectories.

    Supports rewards shaped like:
      - (T, M, N)
      - (M, T, N)
      - (T, M)
      - (M, T)
      - and similar permutations

    Strategy:
      - identify trajectory axis as the axis with size == num_total_trajectories
      - identify time axis as the axis with size == n_steps or n_steps+1 if available
      - sum over assets axis if present
      - sum over time
      - mean over trajectories
    """
    r = np.asarray(rewards)

    if r.ndim == 0:
        raise ValueError(f"Unexpected scalar rewards: rewards.shape={r.shape}")

    # 1) Identify trajectory axis (M)
    traj_axes = [ax for ax, sz in enumerate(r.shape) if sz == num_total_trajectories]
    if len(traj_axes) != 1:
        raise ValueError(
            "Cannot uniquely identify trajectory axis. "
            f"rewards.shape={r.shape}, expected exactly one axis == num_total_trajectories={num_total_trajectories}"
        )
    m_axis = traj_axes[0]

    # 2) Identify time axis (T)
    time_axis = None
    if n_steps is not None:
        cand = [ax for ax, sz in enumerate(r.shape) if ax != m_axis and sz in (n_steps, n_steps + 1)]
        if cand:
            time_axis = cand[0]

    if time_axis is None:
        # fallback: pick the first axis that is not trajectory axis
        time_axis = 0 if m_axis != 0 else (1 if r.ndim > 1 else 0)

    # 3) Asset axis: any remaining axis besides (m_axis, time_axis)
    other_axes = [ax for ax in range(r.ndim) if ax not in (m_axis, time_axis)]
    asset_axis = other_axes[0] if other_axes else None

    # 4) Sum across assets if present and not singleton
    if asset_axis is not None and r.shape[asset_axis] > 1:
        r = r.sum(axis=asset_axis)

        # after summing, axes > asset_axis shift left by 1
        def _shift(ax: int, removed: int) -> int:
            return ax - 1 if ax > removed else ax

        m_axis = _shift(m_axis, asset_axis)
        time_axis = _shift(time_axis, asset_axis)

    # 5) Sum over time -> episode reward per trajectory
    episode = r.sum(axis=time_axis)

    # 6) Mean over trajectories
    baseline_mean = float(np.mean(episode, axis=m_axis))
    return baseline_mean


def estimate_baseline_mean_reward(
    get_env_fn: Callable[..., Any],
    *,
    num_assets: int,
    num_total_trajectories: int = 100_000,
    baseline_fixed_depth: float = 0.5,
    seed: Optional[int] = None,
    **env_kwargs,
) -> float:
    """
    Builds a big env with initial_inventory=0, runs a fixed-depth agent,
    returns mean episode reward (scalar): E[ sum_t sum_assets r_t ].
    """
    big_env = get_env_fn(
        num_trajectories=num_total_trajectories,
        num_assets=num_assets,
        initial_inventory=0,
        **env_kwargs,
    )

    if seed is not None:
        big_env.seed(seed)

    A = int(big_env.action_space.shape[-2])
    agent = _FixedDepthAgent(depth=baseline_fixed_depth, action_shape=(A, 2))

    _, _, rewards = generate_trajectory(big_env, agent)

    n_steps = getattr(big_env, "n_steps", None)
    return _mean_episode_reward_from_rewards(
        rewards,
        num_total_trajectories=num_total_trajectories,
        n_steps=n_steps,
    )


# ============================================================
# 3) Debug printing (consistent & reusable)
# ============================================================
def _print_env_debug(
    env_wrapped: Any,
    *,
    state_indices: Sequence[int],
    num_assets: int,
    do_reward_scaling: bool,
    baseline_mean: Optional[float],
) -> None:
    print("Observation space (after ReduceStateSizeWrapper):")
    print(f"Shape: {env_wrapped.observation_space.shape}")
    print(f"Low bounds: {env_wrapped.observation_space.low}")
    print(f"High bounds: {env_wrapped.observation_space.high}")

    print(f"\nSelected state indices: {list(state_indices)}")

    idx = get_index_map(num_assets)
    print("\nIndex map:")
    for k, v in idx.items():
        print(f"  {k}: {v}")

    spi = env_wrapped.unwrapped.stochastic_process_indices
    print("\nstochastic_process_indices:")
    for k, (lo, hi) in spi.items():
        print(f"  {k}: [{lo}:{hi}] (dim={hi-lo})")

    if do_reward_scaling:
        env_raw = env_wrapped.unwrapped
        print("\nReward scaling:")
        print(f"  baseline_mean: {baseline_mean}")
        print(f"  reward_scaling: {getattr(env_raw, 'reward_scaling', None)}")
        print(f"  normalise_rewards_: {getattr(env_raw, 'normalise_rewards_', None)}")


# ============================================================
# 4) Train-env builder (raw -> reduced obs -> SB wrapper)
# ============================================================
@dataclass
class SbEnvBundle:
    env_raw: Any
    env_wrapped: Any
    sb_env: StableBaselinesTradingEnvironment
    state_indices: List[int]
    baseline_mean: Optional[float]


def make_sb_env(
    get_env_fn: Callable[..., Any],
    *,
    num_trajectories: int,
    num_assets: int,
    initial_inventory: Any = None,
    include_arrivals: bool = True,
    include_lob: bool = True,
    do_reward_scaling: bool = False,
    baseline_num_total_trajectories: int = 100_000,
    baseline_fixed_depth: float = 0.5,
    baseline_seed: Optional[int] = None,
    env_seed: Optional[int] = None,
    debug: bool = True,
    **env_kwargs,
) -> SbEnvBundle:
    """
    Builds raw env, optionally calibrates reward scaling, wraps observation,
    converts to SB environment, and prints consistent debug info.

    Notes on seeding:
      - get_env_fn may already bake in train_seed internally
      - env_seed lets you override/force a seed call on the created env
      - baseline_seed seeds the big baseline env used for calibration
      - env_kwargs are forwarded to get_env_fn (e.g. cross_asset_influence)
    """
    # 1) Build raw env
    try:
        env_raw = get_env_fn(
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            initial_inventory=initial_inventory,
            **env_kwargs,
        )
    except TypeError:
        env_raw = get_env_fn(
            num_trajectories=num_trajectories,
            num_assets=num_assets,
            **env_kwargs,
        )

    if env_seed is not None:
        env_raw.seed(env_seed)

    # 2) Optional reward scaling calibration
    baseline_mean = None
    if do_reward_scaling:
        baseline_mean = estimate_baseline_mean_reward(
            get_env_fn,
            num_assets=num_assets,
            num_total_trajectories=baseline_num_total_trajectories,
            baseline_fixed_depth=baseline_fixed_depth,
            seed=baseline_seed,
            **env_kwargs,
        )
        env_raw.reward_scaling = 1.0 / abs(baseline_mean)
        env_raw.normalise_rewards_ = True

    # 3) Robust state selection
    state_indices = build_state_indices(
        env_raw,
        num_assets=num_assets,
        include_arrivals=include_arrivals,
        include_lob=include_lob,
    )

    # 4) Wrap + VecEnv conversion
    env_wrapped = ReduceStateSizeWrapper(env_raw, list_of_state_indices=state_indices)
    sb_env = StableBaselinesTradingEnvironment(trading_env=env_wrapped)

    # 5) Debug prints
    if debug:
        _print_env_debug(
            env_wrapped,
            state_indices=state_indices,
            num_assets=num_assets,
            do_reward_scaling=do_reward_scaling,
            baseline_mean=baseline_mean,
        )

    return SbEnvBundle(
        env_raw=env_raw,
        env_wrapped=env_wrapped,
        sb_env=sb_env,
        state_indices=state_indices,
        baseline_mean=baseline_mean,
    )


# ============================================================
# 5) Eval-env builder (mirrors train bundle)
# ============================================================
@dataclass
class EvalEnvBundle:
    eval_env_raw: Any
    eval_env_wrapped: Any
    sb_eval_env: StableBaselinesTradingEnvironment


def make_eval_sb_env(
    get_env_fn: Callable[..., Any],
    *,
    train_bundle: SbEnvBundle,
    eval_seed: int = 456,
    eval_num_trajectories: Optional[int] = None,
    initial_inventory: Any = None,
    debug: bool = True,
    **env_kwargs,
) -> EvalEnvBundle:
    """
    Build a fresh evaluation env that mirrors the training setup:
      - same num_assets
      - same reduced observation indices (state_indices)
      - same reward scaling flags (reward_scaling, normalise_rewards_)

    Why:
      - avoids notebook globals (num_trajectories not defined)
      - guarantees consistent observation/reward scaling between train and eval
      - env_kwargs are forwarded to get_env_fn (e.g. cross_asset_influence)
    """
    env_raw_train = train_bundle.env_raw
    num_assets = getattr(env_raw_train, "num_assets", None)
    if num_assets is None:
        raise ValueError("train_bundle.env_raw has no attribute 'num_assets'")

    state_indices = train_bundle.state_indices

    if eval_num_trajectories is None:
        eval_num_trajectories = getattr(env_raw_train, "num_trajectories", 1000)

    # Build eval raw env
    try:
        eval_env_raw = get_env_fn(
            num_trajectories=eval_num_trajectories,
            num_assets=num_assets,
            initial_inventory=initial_inventory,
            **env_kwargs,
        )
    except TypeError:
        eval_env_raw = get_env_fn(
            num_trajectories=eval_num_trajectories,
            num_assets=num_assets,
            **env_kwargs,
        )

    # Seed eval env
    if eval_seed is not None:
        eval_env_raw.seed(eval_seed)

    # Mirror reward scaling
    eval_env_raw.reward_scaling = getattr(env_raw_train, "reward_scaling", 1.0)
    eval_env_raw.normalise_rewards_ = getattr(env_raw_train, "normalise_rewards_", False)

    # Apply same state reduction and SB wrapper
    eval_env_wrapped = ReduceStateSizeWrapper(eval_env_raw, list_of_state_indices=state_indices)
    sb_eval_env = StableBaselinesTradingEnvironment(trading_env=eval_env_wrapped)

    if debug:
        print("Eval env ready")
        print("Eval num_trajectories:", eval_num_trajectories)
        print("Eval seed:", eval_seed)
        print("Eval reward_scaling:", getattr(eval_env_raw, "reward_scaling", None))
        print("Eval normalise_rewards_:", getattr(eval_env_raw, "normalise_rewards_", None))
        print("Eval obs shape:", eval_env_wrapped.observation_space.shape)

    return EvalEnvBundle(
        eval_env_raw=eval_env_raw,
        eval_env_wrapped=eval_env_wrapped,
        sb_eval_env=sb_eval_env,
    )