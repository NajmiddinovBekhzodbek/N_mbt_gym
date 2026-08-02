# mbt_gym/agents/MyopicSymmetricActorCriticPolicy.py
# ===================================================
# "Myopic" variant of SymmetricActorCriticPolicy for the multi-asset benchmark.
#
# Purpose (reviewer benchmark):
#   Same fully-coupled environment and SAME parameters as the coupled model, but each
#   asset's quotes are computed using ONLY that asset's own features. This isolates the
#   value of using cross-asset features in the POLICY (myopic vs. fully-coupled), with no
#   change to the environment, dynamics, reward, or calibration.
#
# How myopia is enforced:
#   When computing asset i's action, we ZERO OUT every other asset's entries in the
#   observation (its inventory q_j, its MO intensities lam_j, its LOB depths dep_j, and
#   its price if present), keeping only [q_i, t, price_i?, lam_i(bid,ask), dep_i(bid,ask)].
#   Time t is shared. Bid/ask symmetry within asset i is preserved (as in the parent).
#
# Critic:
#   myopic_critic=True  (default) -> the value is the SUM of per-asset values, each
#                                    computed from that asset's own-features view only
#                                    (fully myopic: no cross-asset info anywhere).
#   myopic_critic=False           -> centralized (full-state) critic, myopic actor only
#                                    ("centralized training, decentralized execution").
#
# Everything else (observation layout, action packing, symmetry transform) is inherited
# from SymmetricActorCriticPolicy.

from __future__ import annotations

import torch as th

from mbt_gym.agents.SymmetricPPOPolicy import SymmetricActorCriticPolicy


class MyopicSymmetricActorCriticPolicy(SymmetricActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        myopic_critic: bool = True,
        **kwargs,
    ):
        # store BEFORE super().__init__ (network build does not need it, but forward() does)
        self.myopic_critic = bool(myopic_critic)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    # ------------------------------------------------------------------
    # Multiplicative keep-mask: 1 for asset i's own features + time, 0 elsewhere.
    # Mirrors the observation layouts supported by the parent policy.
    #   lam/dep blocks are asset-major: [a0_bid,a0_ask,a1_bid,a1_ask,...]
    # ------------------------------------------------------------------
    def _own_feature_mask(self, D: int, i: int, device, dtype) -> th.Tensor:
        A = self.A
        m = th.zeros(D, device=device, dtype=dtype)
        m[i] = 1.0            # own inventory q_i
        m[A] = 1.0            # time t (shared)
        off = A + 1

        if D == A + 1:                       # [q(A), t]
            return m
        if D == 3 * A + 1:                   # [q(A), t, lam(2A)]
            m[off + 2 * i] = 1.0; m[off + 2 * i + 1] = 1.0
            return m
        if D == 4 * A + 1:                   # [q(A), t, price(A), lam(2A)]
            m[off + i] = 1.0                 # price_i
            loff = off + A
            m[loff + 2 * i] = 1.0; m[loff + 2 * i + 1] = 1.0
            return m
        if D == 5 * A + 1:                   # [q(A), t, lam(2A), dep(2A)]  <-- 2-asset base
            m[off + 2 * i] = 1.0; m[off + 2 * i + 1] = 1.0
            doff = off + 2 * A
            m[doff + 2 * i] = 1.0; m[doff + 2 * i + 1] = 1.0
            return m
        if D == 6 * A + 1:                   # [q(A), t, price(A), lam(2A), dep(2A)]
            m[off + i] = 1.0
            loff = off + A
            m[loff + 2 * i] = 1.0; m[loff + 2 * i + 1] = 1.0
            doff = loff + 2 * A
            m[doff + 2 * i] = 1.0; m[doff + 2 * i + 1] = 1.0
            return m
        raise ValueError(
            f"Unsupported obs_dim={D} for A={A}. Expected {A+1}, {3*A+1}, {4*A+1}, {5*A+1}, or {6*A+1}."
        )

    def _myopic_view_for_asset(self, obs: th.Tensor, i: int) -> th.Tensor:
        obs = obs.float()
        B, D = obs.shape
        mask = self._own_feature_mask(D, i, obs.device, obs.dtype)  # (D,)
        return obs * mask.unsqueeze(0)                              # (B,D), others zeroed

    # ------------------------------------------------------------------
    # ACTOR: asset i's bid and ask depend ONLY on asset i's own features.
    # (overrides the parent; get_distribution() inherited from parent uses this)
    # ------------------------------------------------------------------
    def _mean_actions_flat(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.float()
        B = obs.shape[0]
        A = self.A

        delta_minus = th.empty((B, A), device=obs.device, dtype=obs.dtype)
        delta_plus = th.empty((B, A), device=obs.device, dtype=obs.dtype)

        for i in range(A):
            obs_i = self._myopic_view_for_asset(obs, i)                # own features only
            dm_i = self.action_net_minus(self._latent_pi_from_obs(obs_i))   # (B,A)
            delta_minus[:, i] = dm_i[:, i]

            # bid/ask symmetry within asset i (parent transform; other assets are already 0)
            obs_i_sym = self._symmetrize_obs_for_asset(obs_i, i)
            dp_i = self.action_net_minus(self._latent_pi_from_obs(obs_i_sym))  # (B,A)
            delta_plus[:, i] = dp_i[:, i]

        return self._pack_actions_flat(delta_minus, delta_plus)        # (B,2A)

    # ------------------------------------------------------------------
    # CRITIC
    # ------------------------------------------------------------------
    def _myopic_values(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.float()
        values = None
        for i in range(self.A):
            obs_i = self._myopic_view_for_asset(obs, i)
            v_i = self.value_net(self._latent_vf_from_obs(obs_i))      # (B,1)
            values = v_i if values is None else values + v_i
        return values

    def _values(self, obs: th.Tensor) -> th.Tensor:
        if self.myopic_critic:
            return self._myopic_values(obs)
        return self.value_net(self._latent_vf_from_obs(obs))           # centralized critic

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        return self._values(obs.float())

    # ------------------------------------------------------------------
    # SB3 hooks that must return values (reuse parent's action logic via get_distribution)
    # ------------------------------------------------------------------
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        obs = obs.float()
        dist = self.get_distribution(obs)
        actions_flat = dist.get_actions(deterministic=deterministic)   # (B,2A)
        log_prob = dist.log_prob(actions_flat)
        values = self._values(obs)
        actions = self._unpack_actions_flat(actions_flat)              # (B,A,2)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        obs = obs.float()
        if actions.ndim == 3:
            dm = actions[:, :, 0]
            dp = actions[:, :, 1]
            actions = self._pack_actions_flat(dm, dp)
        elif actions.ndim != 2:
            raise ValueError(f"Unexpected actions shape: {tuple(actions.shape)}")

        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self._values(obs)
        return values, log_prob, entropy
