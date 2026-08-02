import gym
import numpy as np

from mbt_gym.gym.N_index_names import get_index_map  # for multi-asset state indexing


class ReduceStateSizeWrapper(gym.Wrapper):
    """
    Reduces observation space to a subset of indices.
    If no indices are provided, selects inventory (all assets) + time.
    """

    def __init__(self, env, list_of_state_indices: list = None):
        super(ReduceStateSizeWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        idx = get_index_map(env.num_assets)
        if list_of_state_indices is None:
            inventory_range = list(range(idx["INVENTORY_INDEX"], idx["TIME_INDEX"]))
            list_of_state_indices = inventory_range + [idx["TIME_INDEX"]]

        self.list_of_state_indices = list_of_state_indices

        # Create reduced observation space
        obs_low = env.observation_space.low[self.list_of_state_indices]
        obs_high = env.observation_space.high[self.list_of_state_indices]
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        assert obs.ndim == 2, f"Expected batched obs of shape (N, D), got {obs.shape}"
        return obs[:, self.list_of_state_indices]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert obs.ndim == 2, f"Expected batched obs of shape (N, D), got {obs.shape}"
        return obs[:, self.list_of_state_indices], reward, done, info

    @property
    def spec(self):
        return self.env.spec


class NormaliseASObservation(gym.Wrapper):
    """
    Linearly normalizes the observation space to [-1, 1].
    """

    def __init__(self, env):
        super(NormaliseASObservation, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        self.normalisation_factor = 2 / (env.observation_space.high - env.observation_space.low)
        self.normalisation_offset = (env.observation_space.high + env.observation_space.low) / 2

        self.observation_space = gym.spaces.Box(
            low=-np.ones(env.observation_space.shape, dtype=np.float32),
            high=np.ones(env.observation_space.shape, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self):
        obs = self.env.reset()
        return (obs - self.normalisation_offset) * self.normalisation_factor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (obs - self.normalisation_offset) * self.normalisation_factor, reward, done, info


class RemoveTerminalRewards(gym.Wrapper):
    """
    Scales down terminal rewards to smooth training signal near the end of episode.
    Works well when using CjMmCriterion.
    """

    def __init__(self, env, num_final_steps: int = 5):
        super(RemoveTerminalRewards, self).__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if done:
            # Make sure env.reward_function is CjMmCriterion
            reward *= (
                self.env.reward_function.per_step_inventory_aversion
                / self.env.reward_function.terminal_inventory_aversion
            )
        return state, reward, done, info
