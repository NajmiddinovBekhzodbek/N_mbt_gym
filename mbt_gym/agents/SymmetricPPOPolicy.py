# mbt_gym/agents/SymmetricPPOPolicy.py
# ===================================
# SB3-compatible symmetric Actor-Critic policy for mbt_gym multi-asset market making.
#
# Symmetry enforced (per asset i):
#   delta_plus_i(obs) = delta_minus_i(S_i(obs))
# where S_i flips ONLY q_i and swaps ONLY bid/ask features for asset i
# inside the lambda/depth blocks (and optionally price blocks if you ever include them).
#
# This version is designed to match your current reduced observation layout:
#   Minimal:               [q1..qA, t]                               dim = A + 1
#   Lam-only:              [q1..qA, t, lam(2A)]                       dim = 3A + 1
#   Extended (no price):   [q1..qA, t, lam(2A), dep(2A)]              dim = 5A + 1
#   Price + Lam-only:      [q1..qA, t, price(A), lam(2A)]             dim = 4A + 1
#   Extended (with price): [q1..qA, t, price(A), lam(2A), dep(2A)]    dim = 6A + 1
#
# ACTION SHAPE:
#   env action_space.shape = (A, 2)
#   action[..., 0] = delta_minus (bid)
#   action[..., 1] = delta_plus  (ask)
#
# IMPORTANT:
# - SB3 DiagGaussian internally uses a flat vector of size 2A.
# - We explicitly interleave in flat form:
#     [a0_bid, a0_ask, a1_bid, a1_ask, ...]
#
# NOTE on "symmetry during training":
# - PPO samples actions with independent Gaussian noise per dimension.
# - This enforces symmetry in the *mean* of the policy distribution (deterministic contours),
#   not necessarily in every stochastic sample.
# - Your contour plots use deterministic=True, so they should be symmetric if the mapping is correct.

from __future__ import annotations

import torch as th
import torch.nn as nn

try:
    from gymnasium import spaces
except Exception:  # pragma: no cover
    from gym import spaces  # type: ignore

from stable_baselines3.common.policies import ActorCriticPolicy


class SymmetricActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        a_shape = getattr(action_space, "shape", None)
        if a_shape is None or len(a_shape) != 2 or int(a_shape[1]) != 2:
            raise ValueError(f"Expected action_space.shape == (A, 2), got {a_shape}")
        self.A = int(a_shape[0])
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    # ---------------------------------------------------------------------
    # Build networks: SB3 builds feature extractor + mlp_extractor + value_net.
    # We add our custom action head that outputs ONLY delta_minus per asset.
    # ---------------------------------------------------------------------
    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net_minus = nn.Linear(latent_dim_pi, self.A)

    # ---------------------------------------------------------------------
    # Explicit flat <-> (A,2) packing to avoid any ordering ambiguity.
    # Flat ordering we enforce:
    #   [a0_bid, a0_ask, a1_bid, a1_ask, ...]
    # ---------------------------------------------------------------------
    def _pack_actions_flat(self, delta_minus: th.Tensor, delta_plus: th.Tensor) -> th.Tensor:
        """
        delta_minus: (B, A)
        delta_plus : (B, A)
        returns (B, 2A) interleaved: [a0_bid,a0_ask,a1_bid,a1_ask,...]
        """
        if delta_minus.ndim != 2 or delta_plus.ndim != 2:
            raise ValueError("delta_minus and delta_plus must be 2D tensors (B, A)")
        B, A = delta_minus.shape
        if A != self.A or delta_plus.shape != (B, A):
            raise ValueError(f"Shape mismatch: expected (B,{self.A})")
        out = th.empty((B, 2 * A), device=delta_minus.device, dtype=delta_minus.dtype)
        out[:, 0::2] = delta_minus
        out[:, 1::2] = delta_plus
        return out

    def _unpack_actions_flat(self, actions_flat: th.Tensor) -> th.Tensor:
        """
        actions_flat: (B, 2A) interleaved
        returns: (B, A, 2) where [:,:,0]=bid, [:,:,1]=ask
        """
        if actions_flat.ndim != 2 or actions_flat.shape[1] != 2 * self.A:
            raise ValueError(f"Expected actions_flat shape (B,{2*self.A}), got {tuple(actions_flat.shape)}")
        B = actions_flat.shape[0]
        out = th.empty((B, self.A, 2), device=actions_flat.device, dtype=actions_flat.dtype)
        out[:, :, 0] = actions_flat[:, 0::2]
        out[:, :, 1] = actions_flat[:, 1::2]
        return out

    # ---------------------------------------------------------------------
    # Helpers to compute latent features the SB3 way (supports custom feature extractors)
    # ---------------------------------------------------------------------
    def _latent_pi_from_obs(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return latent_pi

    def _latent_vf_from_obs(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return latent_vf

    # ---------------------------------------------------------------------
    # Symmetry transform S_i(obs)
    #
    # Assumptions about reduced obs ordering:
    #  (a) minimal: [q1..qA, t]                     => D = A+1
    #  (b) lam-only: [q1..qA, t, lam(2A)]           => D = 3A+1  (Hawkes + no depth)
    #  (c) extended: [q1..qA, t, lam(2A), dep(2A)]  => D = 5A+1
    #  (d) price+lam-only: [q1..qA, t, price(A), lam(2A)] => D = 4A+1
    #  (e) extended with price: [q1..qA, t, price(A), lam(2A), dep(2A)] => D = 6A+1
    #
    # lam/dep MUST be asset-major: [a0_bid,a0_ask,a1_bid,a1_ask,...]
    #
    # If your lam/dep are stored side-major instead (all bids then all asks),
    # tell me and I’ll give you the swap logic for that layout.
    # ---------------------------------------------------------------------
    def _symmetrize_obs_for_asset(self, obs: th.Tensor, asset_i: int) -> th.Tensor:
        obs = obs.float()
        B, D = obs.shape
        A = self.A
        i = int(asset_i)
        if not (0 <= i < A):
            raise ValueError(f"asset_i must be in [0,{A-1}] got {i}")

        # minimal: [q(A), t]
        if D == A + 1:
            q = obs[:, :A].clone()
            t = obs[:, A:A + 1]
            q[:, i] = -q[:, i]
            return th.cat([q, t], dim=1)

        # swap function for a block shaped (B,2A) with asset-major bid/ask
        def swap_bid_ask_asset_major(x: th.Tensor) -> th.Tensor:
            x = x.clone()
            b = 2 * i
            a = 2 * i + 1
            xb = x[:, b].clone()
            xa = x[:, a].clone()
            x[:, b] = xa
            x[:, a] = xb
            return x

        # lam-only: [q(A), t, lam(2A)] => D = 3A+1
        if D == 3 * A + 1:
            q = obs[:, :A].clone()
            t = obs[:, A:A + 1]
            lam = obs[:, A + 1: A + 1 + 2 * A]

            q[:, i] = -q[:, i]
            lam = swap_bid_ask_asset_major(lam)
            return th.cat([q, t, lam], dim=1)

        # extended no price: [q(A), t, lam(2A), dep(2A)] => D = 5A+1
        if D == 5 * A + 1:
            q = obs[:, :A].clone()
            t = obs[:, A:A + 1]
            lam = obs[:, A + 1: A + 1 + 2 * A]
            dep = obs[:, A + 1 + 2 * A: A + 1 + 4 * A]

            q[:, i] = -q[:, i]
            lam = swap_bid_ask_asset_major(lam)
            dep = swap_bid_ask_asset_major(dep)
            return th.cat([q, t, lam, dep], dim=1)

        # price + lam-only: [q(A), t, price(A), lam(2A)] => D = 4A+1
        if D == 4 * A + 1:
            q = obs[:, :A].clone()
            t = obs[:, A:A + 1]
            price = obs[:, A + 1: A + 1 + A]  # unchanged
            lam = obs[:, A + 1 + A: A + 1 + A + 2 * A]

            q[:, i] = -q[:, i]
            lam = swap_bid_ask_asset_major(lam)
            return th.cat([q, t, price, lam], dim=1)

        # extended with price: [q(A), t, price(A), lam(2A), dep(2A)] => D = 6A+1
        if D == 6 * A + 1:
            q = obs[:, :A].clone()
            t = obs[:, A:A + 1]
            price = obs[:, A + 1: A + 1 + A]  # unchanged
            lam = obs[:, A + 1 + A: A + 1 + A + 2 * A]
            dep = obs[:, A + 1 + A + 2 * A: A + 1 + A + 4 * A]

            q[:, i] = -q[:, i]
            lam = swap_bid_ask_asset_major(lam)
            dep = swap_bid_ask_asset_major(dep)
            return th.cat([q, t, price, lam, dep], dim=1)

        raise ValueError(
            f"Unsupported obs_dim={D} for A={A}. Expected {A+1}, {3*A+1}, {4*A+1}, {5*A+1}, or {6*A+1}."
        )

    # ---------------------------------------------------------------------
    # Compute symmetric MEAN action vector (flat) for the Gaussian distribution.
    # ---------------------------------------------------------------------
    def _mean_actions_flat(self, obs: th.Tensor) -> th.Tensor:
        obs = obs.float()
        B = obs.shape[0]
        A = self.A

        # delta_minus = f(obs)
        latent_pi = self._latent_pi_from_obs(obs)
        delta_minus = self.action_net_minus(latent_pi)  # (B,A)

        # delta_plus_i = delta_minus_i(S_i(obs))
        delta_plus = th.empty((B, A), device=obs.device, dtype=delta_minus.dtype)
        for i in range(A):
            obs_sym = self._symmetrize_obs_for_asset(obs, i)
            latent_pi_sym = self._latent_pi_from_obs(obs_sym)
            dm_sym = self.action_net_minus(latent_pi_sym)  # (B,A)
            delta_plus[:, i] = dm_sym[:, i]

        return self._pack_actions_flat(delta_minus, delta_plus)  # (B,2A)

    # ---------------------------------------------------------------------
    # IMPORTANT SB3 HOOK:
    # SB3 calls _get_action_dist_from_latent(latent_pi) internally to build the action distribution.
    # We override it so that SB3 ALWAYS uses our symmetric mean.
    #
    # latent_pi is computed from obs already by SB3, but we recompute symmetric mean from obs
    # because symmetry depends on S_i(obs).
    # ---------------------------------------------------------------------
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        # SB3 gives us latent_pi, but we need obs to build symmetric mean.
        # The clean way: ignore latent_pi and reconstruct mean from self._last_obs.
        # However SB3 does not expose obs here reliably.
        #
        # So instead we override get_distribution() below which *does* receive obs.
        # This method remains as fallback; SB3 will still call get_distribution(obs) in most paths.
        mean_actions = self.action_net(latent_pi)  # default, unused ideally
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def get_distribution(self, obs: th.Tensor):
        mean_actions_flat = self._mean_actions_flat(obs)  # (B,2A)
        return self.action_dist.proba_distribution(mean_actions_flat, self.log_std)

    # ---------------------------------------------------------------------
    # forward(): used during rollouts
    # MUST return actions shaped (B,A,2) so np.clip broadcasts with low/high (A,2).
    # ---------------------------------------------------------------------
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        obs = obs.float()

        dist = self.get_distribution(obs)
        actions_flat = dist.get_actions(deterministic=deterministic)  # (B,2A)
        log_prob = dist.log_prob(actions_flat)

        latent_vf = self._latent_vf_from_obs(obs)
        values = self.value_net(latent_vf)

        actions = self._unpack_actions_flat(actions_flat)  # (B,A,2)
        return actions, values, log_prob

    # ---------------------------------------------------------------------
    # predict hook
    # ---------------------------------------------------------------------
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        observation = observation.float()
        dist = self.get_distribution(observation)
        actions_flat = dist.get_actions(deterministic=deterministic)
        return self._unpack_actions_flat(actions_flat)

    # ---------------------------------------------------------------------
    # PPO training hook
    # ---------------------------------------------------------------------
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        obs = obs.float()

        # Convert actions to flat (B,2A) interleaved if needed
        if actions.ndim == 3:
            dm = actions[:, :, 0]
            dp = actions[:, :, 1]
            actions = self._pack_actions_flat(dm, dp)
        elif actions.ndim == 2:
            # assume already (B,2A)
            pass
        else:
            raise ValueError(f"Unexpected actions shape: {tuple(actions.shape)}")

        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        latent_vf = self._latent_vf_from_obs(obs)
        values = self.value_net(latent_vf)

        return values, log_prob, entropy
