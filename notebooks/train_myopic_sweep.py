#!/usr/bin/env python
# ============================================================
# MYOPIC ρ-SWEEP — headless training driver
# Trains 2-asset MYOPIC agents in the fully-coupled environment for
#   Trial 3 -> rho = rho_c = 1
#   Trial 4 -> rho = rho_c = 2
#   Trial 5 -> rho = rho_c = 3
#   Trial 6 -> rho = rho_c = 4
#   Trial 7 -> rho = rho_c = 5           (rho = trial_num - 2)
# Everything is saved exactly as in the notebook, under the "_Myopic" tag,
# with the correct trial_num. Runs each trial back-to-back (headless).
#
# Run in a TERMINAL (AC power, Dropbox paused):
#   conda activate N_mbt_gym_arm
#   caffeinate -i python notebooks/train_myopic_sweep.py 2>&1 | tee train_myopic_sweep.log
#
# TensorBoard (separate terminal) picks up each run automatically:
#   tensorboard --logdir ./N_tensorboard
#
# Set SMOKE_TEST=True to dry-run (few steps, save NOTHING) — for verification only.
# ============================================================
import os, sys, time, gc, traceback

# --- robust project root ---
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p):
    if os.path.isdir(os.path.join(_p, "mbt_gym")) and os.path.isdir(os.path.join(_p, "environment_set_up")):
        PROJECT_ROOT = _p
        break
    _p = os.path.dirname(_p)
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Working directory:", os.getcwd())

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from mbt_gym.agents.MyopicSymmetricActorCriticPolicy import MyopicSymmetricActorCriticPolicy
from environment_set_up.N_env_setup import make_sb_env, make_eval_sb_env
from environment_set_up.N_callbacks import AccurateEvalCallback, SaveAtTimestepsCallback
from environment_set_up.N_run_logging import save_run_metadata
from environment_set_up.N_env_builders import get_cj_env, set_env_globals

# ============================================================
SMOKE_TEST = os.environ.get("SWEEP_SMOKE", "0") == "1"

# global configs (identical to the notebook)
terminal_time = 1.0
phi = 0.005
alpha = 0.2
n_steps = 100
num_trajectories = 1000
total_timesteps = 100_000_000
network_seed, eval_seed, train_seed = 789, 456, 123
num_assets = 2
do_reward_scaling = False
baseline_fixed_depth = 0.5
initial_inventory = (-1, 2)

# the sweep: (trial_num, rho);  rho applied to BOTH cross and c_cross
SWEEP = [(3, 1.0), (4, 2.0), (5, 3.0), (6, 4.0), (7, 5.0)]

save_steps = [30_000_000, 50_000_000, 70_000_000, 100_000_000]

def lr_scheduler(progress: float) -> float:
    reversed_progress = 1 - progress
    initial_lr, final_lr = 0.0003, 0.00003
    return initial_lr - (initial_lr - final_lr) * reversed_progress

policy_kwargs = dict(net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])

if SMOKE_TEST:
    print("=" * 64); print(" SMOKE TEST — build each trial, verify rho, SAVE NOTHING"); print("=" * 64)


def train_trial(trial_num, rho):
    tag = f"{num_assets}Assets_Myopic"
    run_name = f"PPO_{num_assets}Assets_Myopic_Trial{trial_num}"
    cross = [[0.0, rho], [rho, 0.0]]
    c_cross = [[0.0, rho], [rho, 0.0]]

    print("\n" + "=" * 64)
    print(f" TRIAL {trial_num}  |  rho = rho_c = {rho}  |  {run_name}")
    print("=" * 64)

    set_env_globals(terminal_time=terminal_time, n_steps=n_steps, phi=phi, alpha=alpha, train_seed=train_seed)
    train_bundle = make_sb_env(get_env_fn=get_cj_env, num_trajectories=num_trajectories, num_assets=num_assets,
                               do_reward_scaling=do_reward_scaling, baseline_fixed_depth=baseline_fixed_depth,
                               env_seed=train_seed, debug=False, initial_inventory=initial_inventory,
                               cross_asset_influence=cross, c_cross_asset_influence=c_cross)
    eval_bundle = make_eval_sb_env(get_env_fn=get_cj_env, train_bundle=train_bundle, eval_seed=eval_seed,
                                   initial_inventory=initial_inventory, debug=False,
                                   cross_asset_influence=cross, c_cross_asset_influence=c_cross)
    # confirm the environment actually carries this rho
    import numpy as np
    am = train_bundle.env_raw.model_dynamics.arrival_model
    lm = train_bundle.env_raw.model_dynamics.lob_depth_model
    print("  env arrival cross:", np.asarray(am.cross_asset_influence).tolist(),
          "| lob c_cross:", np.asarray(lm.c_cross_asset_influence).tolist())

    sb_env = VecMonitor(train_bundle.sb_env)
    sb_eval_env = VecMonitor(eval_bundle.sb_eval_env)
    state_indices = train_bundle.state_indices

    tensorboard_logdir = f"./N_tensorboard/PPO_{tag}/{run_name}/"
    best_model_path = f"./N_SB_models/PPO_Best_{tag}/{run_name}"
    ckpt_root = f"./N_SB_models/PPO_Checkpoints_{tag}/{run_name}"

    steps = 2 * n_steps * num_trajectories if SMOKE_TEST else total_timesteps

    if SMOKE_TEST:
        print(f"  [smoke] rho ok, paths -> {tensorboard_logdir} (NOT created), skipping train/save")
        del train_bundle, eval_bundle, sb_env, sb_eval_env
        gc.collect()
        return

    os.makedirs(tensorboard_logdir, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    PPO_params = {
        "policy": MyopicSymmetricActorCriticPolicy, "env": sb_env, "verbose": 1,
        "policy_kwargs": policy_kwargs, "tensorboard_log": tensorboard_logdir,
        "n_epochs": 3, "batch_size": int(n_steps * num_trajectories / 10),
        "n_steps": int(n_steps), "learning_rate": lr_scheduler,
        "ent_coef": 0.0, "seed": network_seed, "gamma": 1.0,
    }
    model = PPO(**PPO_params, device="cpu")

    callback_params = dict(eval_env=sb_eval_env, n_eval_episodes=1000, best_model_save_path=best_model_path,
                           deterministic=True, eval_freq=500, verbose=2)
    best_cb = AccurateEvalCallback(**callback_params, seed=eval_seed)
    save_cb = SaveAtTimestepsCallback(save_steps=save_steps, save_dir=ckpt_root, name_prefix=run_name, verbose=1)

    save_run_metadata(run_name=run_name, tag=tag, total_timesteps=total_timesteps, num_assets=num_assets,
                      num_trajectories=num_trajectories, n_steps=n_steps, terminal_time=terminal_time,
                      phi=phi, alpha=alpha, state_indices=state_indices, train_bundle=train_bundle,
                      model=model, PPO_params=PPO_params, policy_kwargs=policy_kwargs, best_cb=best_cb,
                      callback_params=callback_params, train_seed=train_seed, network_seed=network_seed,
                      eval_seed=eval_seed, baseline_mean_reward=None, baseline_fixed_depth=None)

    t0 = time.time()
    print(f"  training {steps:,} timesteps ...", flush=True)
    model.learn(total_timesteps=steps, callback=[save_cb, best_cb])
    dt = time.time() - t0
    h, rem = divmod(dt, 3600); mm, ss = divmod(rem, 60)
    print(f"  DONE trial {trial_num} in {int(h)}h {int(mm)}m {int(ss)}s | best_mean_reward={best_cb.best_mean_reward}")

    del model, train_bundle, eval_bundle, sb_env, sb_eval_env, best_cb, save_cb
    gc.collect()


if __name__ == "__main__":
    t_all = time.time()
    for trial_num, rho in SWEEP:
        try:
            train_trial(trial_num, rho)
        except Exception:
            print(f"!! TRIAL {trial_num} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"\n===== SWEEP COMPLETE in {(time.time()-t_all)/3600:.2f} h =====")
