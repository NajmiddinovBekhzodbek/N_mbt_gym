# PPO_2Assets_Trial11
- tag: 2Assets
- created_at: 2026-03-11T12:56:36.920182
- total_timesteps: 100000000

## Seeds
{
  "train_seed": 123,
  "network_seed": 789,
  "eval_seed": 456
}

## Environment
{
  "terminal_time": 1.0,
  "n_steps": 100,
  "phi": 0.005,
  "alpha": 0.2,
  "num_assets": 2,
  "num_trajectories": 1000,
  "max_inventory": 100,
  "normalise_action_space": false,
  "normalise_observation_space": false,
  "normalise_rewards": false,
  "reward_scaling": 1.0,
  "baseline_mean_reward": null,
  "baseline_mean_reward_inferred": null,
  "baseline_fixed_depth": null,
  "state_indices": [
    1,
    2,
    3,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13
  ],
  "initial_inventory_spec": "(-1, 2)"
}

## Midprice model
{
  "type": "BrownianMotionMidpriceModel",
  "volatility": [
    0.1,
    0.1
  ],
  "drift": [
    0.0,
    0.0
  ],
  "initial_price": [
    100.0,
    100.0
  ],
  "step_size": 0.01,
  "terminal_time": 1.0
}

## Arrival model
{
  "type": "SynchronousHawkesArrivalModel",
  "baseline_arrival_rate": [
    [
      10.0,
      10.0
    ],
    [
      12.0,
      12.0
    ]
  ],
  "mean_reversion_speed": [
    15.0,
    18.0
  ],
  "self_jump_size": [
    3.0,
    3.2
  ],
  "mutual_jump_size": [
    1.5,
    1.6
  ],
  "synchrony_factor": [
    0.05,
    0.05
  ],
  "cross_asset_influence": [
    [
      0.0,
      7.0
    ],
    [
      7.0,
      0.0
    ]
  ],
  "step_size": 0.01
}

## LOB depth model
{
  "type": "SynchronousLOBDepthModel",
  "c_baseline_depth": [
    [
      2.0,
      2.0
    ],
    [
      2.5,
      2.5
    ]
  ],
  "c_mean_reversion_speed": [
    10.0,
    12.0
  ],
  "c_self_jump_size": [
    0.6,
    0.65
  ],
  "c_mutual_jump_size": [
    0.3,
    0.32
  ],
  "c_synchrony_factor": [
    0.05,
    0.05
  ],
  "c_cross_asset_influence": [
    [
      0.0,
      0.0
    ],
    [
      0.0,
      0.0
    ]
  ],
  "step_size": 0.01
}

## PPO
{
  "policy": "SymmetricActorCriticPolicy",
  "policy_kwargs": "{'net_arch': [{'pi': [256, 256, 256], 'vf': [256, 256, 256]}]}",
  "activation_fn": "Tanh",
  "n_epochs": 3,
  "batch_size": 10000,
  "n_steps": 100,
  "ent_coef": 0.0,
  "gamma": 1.0,
  "gae_lambda": 0.95,
  "clip_range": "<function constant_fn.<locals>.func at 0x0000024CF6030940>",
  "lr_spec": {
    "type": "callable",
    "function_name": "lr_scheduler",
    "initial_lr": 0.0003,
    "final_lr": 3.0000000000000024e-05,
    "source": "def lr_scheduler(progress: float) -> float:\n    reversed_progress = 1 - progress\n    initial_lr = 0.0003\n    final_lr = 0.00003\n    return initial_lr - (initial_lr - final_lr) * reversed_progress\n"
  },
  "device": "cpu"
}