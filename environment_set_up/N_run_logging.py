
from __future__ import annotations

import os
import json
import inspect
import sys
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple, Union


# -------------------------
# Small helpers (safe JSON)
# -------------------------
def tolist(x: Any) -> Any:
    return x.tolist() if hasattr(x, "tolist") else x


def safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def infer_lr_spec(lr: Any) -> Dict[str, Any]:
    """
    Try to infer learning rate schedule spec for logging.
    SB3 uses progress_remaining: 1.0 at start -> 0.0 at end.
    """
    if isinstance(lr, (float, int)):
        return {"type": "fixed", "value": float(lr)}

    if callable(lr):
        name = getattr(lr, "__name__", "callable_lr")
        spec: Dict[str, Any] = {"type": "callable", "function_name": name}
        try:
            spec["initial_lr"] = float(lr(1.0))
            spec["final_lr"] = float(lr(0.0))
        except Exception:
            pass
        try:
            spec["source"] = inspect.getsource(lr)
        except Exception:
            spec["source"] = None
        return spec

    return {"type": "unknown"}


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def csv_safe(v: Any) -> str:
    s = safe_str(v)
    s = s.replace(",", ";")
    s = s.replace("\n", "\\n")
    return s


def _json_default(o: Any) -> Any:
    """
    json.dump default handler: convert unknown objects to strings.
    This prevents errors like "ABCMeta is not JSON serializable".
    """
    return safe_str(o)


# -------------------------
# Public API
# -------------------------
@dataclass
class RunLogPaths:
    run_dir: str
    json_path: str
    csv_path: str
    xlsx_path: Optional[str]
    md_path: str


def save_run_metadata(
    *,
    run_name: str,
    tag: str,
    total_timesteps: int,
    num_assets: int,
    num_trajectories: int,
    n_steps: int,
    terminal_time: float,
    phi: float,
    alpha: float,
    state_indices: Sequence[int],
    # bundles / objects
    train_bundle: Any,          # expects .env_raw (your SbEnvBundle)
    model: Any,                 # SB3 PPO model
    PPO_params: Dict[str, Any],
    policy_kwargs: Dict[str, Any],
    # callbacks (optional, but you use them)
    best_cb: Optional[Any] = None,
    callback_params: Optional[Dict[str, Any]] = None,
    # seeds
    train_seed: Optional[int] = None,
    network_seed: Optional[int] = None,
    eval_seed: Optional[int] = None,
    # baseline logging (optional)
    baseline_mean_reward: Optional[float] = None,
    baseline_fixed_depth: Optional[float] = None,
    # output
    run_dir_root: str = "./N_runs",
    save_excel: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], RunLogPaths]:
    """
    Saves:
      1) JSON metadata
      2) Flattened CSV metadata
      3) Optional Excel (if pandas/openpyxl available)
      4) Human-readable Markdown summary

    Returns:
      (run_meta_dict, RunLogPaths)
    """

    # Try to get base env even if wrapped; you now use bundles
    base_env = getattr(train_bundle, "env_raw", train_bundle)
    md = getattr(base_env, "model_dynamics", None)

    # Pull model components safely
    mid = getattr(md, "midprice_model", None)
    arr = getattr(md, "arrival_model", None)
    lob = getattr(md, "lob_depth_model", None)
    fill = getattr(md, "fill_probability_model", None)

    # Learning rate spec (auto)
    lr_spec = infer_lr_spec(PPO_params.get("learning_rate"))

    # -------------------------
    # 1) Activation function
    # -------------------------
    activation_fn = None
    if isinstance(policy_kwargs, dict) and "activation_fn" in policy_kwargs:
        activation_fn = policy_kwargs["activation_fn"]
    else:
        activation_fn = getattr(getattr(model, "policy", None), "activation_fn", None)

    activation_name = getattr(activation_fn, "__name__", safe_str(activation_fn)) if activation_fn is not None else None

    # -------------------------
    # 2) Initial inventory spec
    # -------------------------
    initial_inventory_spec = getattr(base_env, "initial_inventory", None)
    if hasattr(initial_inventory_spec, "shape"):
        initial_inventory_spec_out = tolist(initial_inventory_spec)
    else:
        initial_inventory_spec_out = safe_str(initial_inventory_spec)

    # -------------------------
    # 3) Baseline reward (scaling)
    # -------------------------
    inferred_baseline_mean = None
    rs = getattr(base_env, "reward_scaling", None)
    if baseline_mean_reward is None and rs not in (None, 0):
        # assumes reward_scaling ≈ 1/abs(baseline_mean)
        try:
            inferred_baseline_mean = 1.0 / float(rs)
        except Exception:
            inferred_baseline_mean = None

    # baseline fixed depth
    baseline_fixed_depth_out = None
    if baseline_fixed_depth is not None:
        try:
            baseline_fixed_depth_out = float(baseline_fixed_depth)
        except Exception:
            baseline_fixed_depth_out = None

    # -------------------------
    # 4) Versions
    # -------------------------
    versions: Dict[str, Any] = {}

    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except Exception:
        versions["numpy"] = None

    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = None

    try:
        import gym
        versions["gym"] = getattr(gym, "__version__", None)
    except Exception:
        versions["gym"] = None

    try:
        import stable_baselines3 as sb3
        versions["stable_baselines3"] = sb3.__version__
    except Exception:
        versions["stable_baselines3"] = None

    versions["python"] = sys.version
    versions["platform"] = platform.platform()

    # -------------------------
    # 5) Evaluation settings
    # -------------------------
    callback_params = callback_params or {}

    evaluation_settings = {
        "callback_class": best_cb.__class__.__name__ if best_cb is not None else None,
        "eval_freq": callback_params.get("eval_freq"),
        "n_eval_episodes": callback_params.get("n_eval_episodes"),
        "deterministic": callback_params.get("deterministic"),
        "eval_seed": eval_seed,
    }

    # -------------------------
    # 6) Baseline policy block
    # -------------------------
    baseline_policy = {
        "type": "fixed_depth",
        "fixed_depth": baseline_fixed_depth_out,
        "applies_to": "all_assets_bid_ask",
    }

    # -------------------------
    # Build metadata dict
    # -------------------------
    run_meta: Dict[str, Any] = {
        "run_name": run_name,
        "tag": tag,
        "created_at": datetime.now().isoformat(),
        "total_timesteps": int(total_timesteps),

        "seeds": {
            "train_seed": train_seed,
            "network_seed": network_seed,
            "eval_seed": eval_seed,
        },

        "versions": versions,
        "evaluation": evaluation_settings,
        "baseline_policy": baseline_policy,

        "env_global": {
            "terminal_time": float(terminal_time),
            "n_steps": int(n_steps),
            "phi": float(phi),
            "alpha": float(alpha),
            "num_assets": int(num_assets),
            "num_trajectories": int(num_trajectories),
            "max_inventory": int(n_steps),
            "normalise_action_space": bool(getattr(base_env, "normalise_action_space_", False)),
            "normalise_observation_space": bool(getattr(base_env, "normalise_observation_space_", False)),
            "normalise_rewards": bool(getattr(base_env, "normalise_rewards_", False)),
            "reward_scaling": float(getattr(base_env, "reward_scaling", 1.0)),
            "baseline_mean_reward": float(baseline_mean_reward) if baseline_mean_reward is not None else None,
            "baseline_mean_reward_inferred": float(inferred_baseline_mean) if inferred_baseline_mean is not None else None,
            "baseline_fixed_depth": baseline_fixed_depth_out,
            "state_indices": tolist(state_indices),
            "initial_inventory_spec": initial_inventory_spec_out,
        },

        "midprice_model": {
            "type": mid.__class__.__name__ if mid is not None else None,
            "volatility": tolist(getattr(mid, "volatility", None)),
            "drift": tolist(getattr(mid, "drift", None)),
            "initial_price": (
                tolist(getattr(mid, "initial_state", None)[0, :, 0])
                if getattr(mid, "initial_state", None) is not None else None
            ),
            "step_size": getattr(mid, "step_size", None),
            "terminal_time": getattr(mid, "terminal_time", None),
        },

        "arrival_model": {
            "type": arr.__class__.__name__ if arr is not None else None,
            "baseline_arrival_rate": tolist(getattr(arr, "baseline_arrival_rate", None)),
            "mean_reversion_speed": tolist(getattr(arr, "mean_reversion_speed", None)),
            "self_jump_size": tolist(getattr(arr, "self_jump_size", None)),
            "mutual_jump_size": tolist(getattr(arr, "mutual_jump_size", None)),
            "synchrony_factor": tolist(getattr(arr, "synchrony_factor", None)),
            "cross_asset_influence": tolist(getattr(arr, "cross_asset_influence", None)),
            "step_size": getattr(arr, "step_size", None),
        },

        "lob_depth_model": {
            "type": lob.__class__.__name__ if lob is not None else None,
            "c_baseline_depth": tolist(getattr(lob, "c_baseline_depth", None)),
            "c_mean_reversion_speed": tolist(getattr(lob, "c_mean_reversion_speed", None)),
            "c_self_jump_size": tolist(getattr(lob, "c_self_jump_size", None)),
            "c_mutual_jump_size": tolist(getattr(lob, "c_mutual_jump_size", None)),
            "c_synchrony_factor": tolist(getattr(lob, "c_synchrony_factor", None)),
            "c_cross_asset_influence": tolist(getattr(lob, "c_cross_asset_influence", None)),
            "step_size": getattr(lob, "step_size", None),
        },

        "fill_probability_model": {
            "type": fill.__class__.__name__ if fill is not None else None,
            "depends_on": "lob_depth_model",
        },

        "reward": {
            "type": base_env.reward_function.__class__.__name__ if hasattr(base_env, "reward_function") else None,
            "per_step_inventory_aversion": float(phi),
            "terminal_inventory_aversion": float(alpha),
            "inventory_exponent": 2.0,
        },

        "ppo": {
            "policy": getattr(PPO_params.get("policy"), "__name__", safe_str(PPO_params.get("policy"))),
            # policy_kwargs can contain non-JSON objects; store string for robustness
            "policy_kwargs": safe_str(policy_kwargs),
            "activation_fn": activation_name,
            "n_epochs": PPO_params.get("n_epochs"),
            "batch_size": PPO_params.get("batch_size"),
            "n_steps": PPO_params.get("n_steps"),
            "ent_coef": PPO_params.get("ent_coef"),
            "gamma": getattr(model, "gamma", None),
            "gae_lambda": getattr(model, "gae_lambda", None),
            "clip_range": safe_str(getattr(model, "clip_range", None)),
            "lr_spec": lr_spec,
            "device": safe_str(getattr(model, "device", "cpu")),
        },
    }

    # -------------------------
    # Save outputs
    # -------------------------
    run_dir = os.path.join(run_dir_root, tag, run_name)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, f"{run_name}_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=_json_default)

    flat = flatten_dict(run_meta)
    csv_path = os.path.join(run_dir, f"{run_name}_meta.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("key,value\n")
        for k, v in flat.items():
            f.write(f"{k},{csv_safe(v)}\n")

    xlsx_path = None
    if save_excel:
        try:
            import pandas as pd  # type: ignore
            df = pd.DataFrame(list(flat.items()), columns=["key", "value"])
            xlsx_path = os.path.join(run_dir, f"{run_name}_meta.xlsx")
            df.to_excel(xlsx_path, index=False)
        except Exception:
            xlsx_path = None

    # Markdown summary
    summary_lines = [
        f"# {run_name}",
        f"- tag: {tag}",
        f"- created_at: {run_meta['created_at']}",
        f"- total_timesteps: {run_meta['total_timesteps']}",
        "",
        "## Seeds",
        json.dumps(run_meta["seeds"], indent=2, default=_json_default),
        "",
        "## Environment",
        json.dumps(run_meta["env_global"], indent=2, default=_json_default),
        "",
        "## Midprice model",
        json.dumps(run_meta["midprice_model"], indent=2, default=_json_default),
        "",
        "## Arrival model",
        json.dumps(run_meta["arrival_model"], indent=2, default=_json_default),
        "",
        "## LOB depth model",
        json.dumps(run_meta["lob_depth_model"], indent=2, default=_json_default),
        "",
        "## PPO",
        json.dumps(run_meta["ppo"], indent=2, default=_json_default),
    ]
    md_path = os.path.join(run_dir, f"{run_name}_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    paths = RunLogPaths(
        run_dir=run_dir,
        json_path=json_path,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        md_path=md_path,
    )

    if verbose:
        print(f"Saved run metadata to:\n  {json_path}\n  {csv_path}")
        if xlsx_path is not None:
            print(f"Also saved Excel:\n  {xlsx_path}")
        print(f"Also saved summary Markdown:\n  {md_path}")

    return run_meta, paths
