#!/usr/bin/env python
# ============================================================
# MULTI-SEED reproducibility runs of the 2-asset BASE model (Trial1)
# ------------------------------------------------------------
# Retrains the EXACT Trial1 fully-coupled configuration (rho = rho_c = 0.05, 100M
# timesteps, identical hyperparameters) with 4 additional seeds, to answer the
# reviewer's request to report results across multiple seeds (convergence/robustness).
#
# Base model reference (already trained, counts as the 5th seed):
#   PPO_2Assets_Trial1   (checkpoint PPO_2Assets_Trial1_100M.zip;
#                         metadata in N_runs/2Assets/PPO_2Assets_Trial1/)
#   seeds used by the base: network=789, train=123, eval=456
#
# For each master seed s in {2, 3, 4, 5} the run is FULLY INDEPENDENT (three distinct
# seeds in disjoint ranges, and eval_seed != train_seed so evaluation stays
# OUT-OF-SAMPLE):
#       network_seed = 1000 + s      (1002 .. 1005)
#       train_seed   = 2000 + s      (2002 .. 2005)
#       eval_seed    = 3000 + s      (3002 .. 3005)
#   -> set FIX_EVAL_SEED = True to instead hold eval_seed = 456 for every run
#      (matches the base; gives the tightest "training-variance-only" curve band).
#
# Everything is saved EXACTLY like N_Bek_Learning_...py (same dirs / tag / metadata),
# with "_seed{s}" appended to the run name, so NOTHING collides with PPO_2Assets_Trial1:
#       run_name    = PPO_2Assets_Trial1_seed{s}
#       checkpoints -> N_SB_models/PPO_Checkpoints_2Assets/PPO_2Assets_Trial1_seed{s}/
#       best model  -> N_SB_models/PPO_Best_2Assets/PPO_2Assets_Trial1_seed{s}/
#       tensorboard -> N_tensorboard/PPO_2Assets/PPO_2Assets_Trial1_seed{s}/
#       metadata    -> N_runs/2Assets/PPO_2Assets_Trial1_seed{s}/
# Only ONE checkpoint is saved, at 100M (save_steps = [100_000_000]). The eval callback
# still runs (eval_freq=500) so you get the training curve, and it keeps a single
# best_model.zip (overwritten on improvement) -- no many-checkpoint clutter.
#
# Run in a TERMINAL (AC power, Dropbox paused), native-arm env for speed:
#   conda activate N_mbt_gym_arm
#   caffeinate -i python notebooks/Learning_with_different_seeds.py 2>&1 | tee learning_seeds.log
#
# TensorBoard (separate terminal) picks up each run automatically:
#   tensorboard --logdir ./N_tensorboard
#
# Set SWEEP_SMOKE=1 to dry-run (build each run, verify seeds/paths/env, SAVE NOTHING).
# ============================================================
import os, sys, time, gc, traceback

# --- robust project root (works no matter where it's launched from) ---
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
from mbt_gym.agents.SymmetricPPOPolicy import SymmetricActorCriticPolicy      # full-state (coupled) base policy
from environment_set_up.N_env_setup import make_sb_env, make_eval_sb_env
from environment_set_up.N_callbacks import AccurateEvalCallback, SaveAtTimestepsCallback
from environment_set_up.N_run_logging import save_run_metadata
from environment_set_up.N_env_builders import get_cj_env, set_env_globals

# ============================================================
SMOKE_TEST = os.environ.get("SWEEP_SMOKE", "0") == "1"
FIX_EVAL_SEED = False     # False -> independent eval market per run (eval_seed = 3000+s)
                          # True  -> hold eval_seed = 456 for all runs (matches base)

# --- global configs (IDENTICAL to the Trial1 base model, from its saved metadata) ---
terminal_time = 1.0
phi = 0.005
alpha = 0.2
n_steps = 100
num_trajectories = 1000
total_timesteps = 100_000_000
num_assets = 2
do_reward_scaling = False
baseline_fixed_depth = 0.5              # only matters if do_reward_scaling=True
initial_inventory = (-1, 2)

# Trial1 dynamics: cross-asset influence = 0.05 for BOTH MO intensities and LOB depths
rho = 0.05
cross   = [[0.0, rho], [rho, 0.0]]
c_cross = [[0.0, rho], [rho, 0.0]]

# the additional seeds to run (the base Trial1 is the 5th seed, already trained)
MASTER_SEEDS = [2, 3, 4, 5]

# only ONE checkpoint, saved at 100M
save_steps = [100_000_000]

def seeds_for(s):
    network_seed = 1000 + s
    train_seed   = 2000 + s
    eval_seed    = 456 if FIX_EVAL_SEED else 3000 + s
    return network_seed, train_seed, eval_seed

def lr_scheduler(progress: float) -> float:
    reversed_progress = 1 - progress
    initial_lr, final_lr = 0.0003, 0.00003
    return initial_lr - (initial_lr - final_lr) * reversed_progress

policy_kwargs = dict(net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])

if SMOKE_TEST:
    print("=" * 64)
    print(" SMOKE TEST — build each seed run, verify seeds/paths/env, SAVE NOTHING")
    print("=" * 64)


def train_one_seed(s):
    network_seed, train_seed, eval_seed = seeds_for(s)
    tag = f"{num_assets}Assets"
    run_name = f"PPO_{num_assets}Assets_Trial1_seed{s}"

    print("\n" + "=" * 64)
    print(f" SEED {s}  |  network={network_seed}  train={train_seed}  eval={eval_seed}  |  {run_name}")
    print("=" * 64)

    set_env_globals(terminal_time=terminal_time, n_steps=n_steps, phi=phi, alpha=alpha, train_seed=train_seed)
    train_bundle = make_sb_env(get_env_fn=get_cj_env, num_trajectories=num_trajectories, num_assets=num_assets,
                               do_reward_scaling=do_reward_scaling, baseline_fixed_depth=baseline_fixed_depth,
                               env_seed=train_seed, debug=False, initial_inventory=initial_inventory,
                               cross_asset_influence=cross, c_cross_asset_influence=c_cross)
    eval_bundle = make_eval_sb_env(get_env_fn=get_cj_env, train_bundle=train_bundle, eval_seed=eval_seed,
                                   initial_inventory=initial_inventory, debug=False,
                                   cross_asset_influence=cross, c_cross_asset_influence=c_cross)

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
        print(f"  [smoke] seeds/paths/env ok -> {run_name} (dirs NOT created), skipping train/save")
        del train_bundle, eval_bundle, sb_env, sb_eval_env
        gc.collect()
        return

    os.makedirs(tensorboard_logdir, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    PPO_params = {
        "policy": SymmetricActorCriticPolicy, "env": sb_env, "verbose": 1,
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
    print(f"  DONE seed {s} in {int(h)}h {int(mm)}m {int(ss)}s | best_mean_reward={best_cb.best_mean_reward}")

    del model, train_bundle, eval_bundle, sb_env, sb_eval_env, best_cb, save_cb
    gc.collect()


if __name__ == "__main__":
    t_all = time.time()
    for s in MASTER_SEEDS:
        try:
            train_one_seed(s)
        except Exception:
            print(f"!! SEED {s} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"\n===== MULTI-SEED RUN COMPLETE in {(time.time()-t_all)/3600:.2f} h =====")
