from collections import OrderedDict
from copy import copy, deepcopy
from typing import Union, Tuple, Callable

import gym
import numpy as np
from gym.spaces import Box

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.N_Bek_ModelDynamics import Bek_ModelDynamics, Bek_LimitOrderModelDynamics
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.stochastic_processes.N_StochasticProcessModel import StochasticProcessModel
from mbt_gym.stochastic_processes.N_arrival_models import ArrivalModel, SynchronousHawkesArrivalModel, SynchronousLOBDepthModel
from mbt_gym.stochastic_processes.N_fill_probability_models import FillProbabilityModel, DynamicLOBExponentialFillFunction
from mbt_gym.stochastic_processes.N_midprice_models import MidpriceModel, BrownianMotionMidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel
from mbt_gym.gym.info_calculators import InfoCalculator
from mbt_gym.rewards.N_RewardFunctions import RewardFunction, PnL

from mbt_gym.gym.N_index_names import get_index_map



class Bek_TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 200,
        reward_function: RewardFunction = None,
        model_dynamics: Bek_ModelDynamics = None,
        initial_cash: float = 0.0,
        initial_inventory: Union[int, Tuple[float, float], np.ndarray] = None,
        max_inventory: int = 10_000,
        max_cash: float = None,
        max_stock_price: float = None,
        start_time: Union[float, int, Callable] = 0.0,
        info_calculator: InfoCalculator = None,
        seed: int = None,
        num_trajectories: int = 1,
        num_assets: int = 1,
        normalise_action_space: bool = True,
        normalise_observation_space: bool = True,
        normalise_rewards: bool = False,
    ):
        super(Bek_TradingEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.num_assets = num_assets
        self.index_map = get_index_map(self.num_assets)
        self._step_size = self.terminal_time / self.n_steps
        self.reward_function = reward_function or PnL(num_assets=num_assets)
        self._num_trajectories = num_trajectories  # initialize the backing variable early
        self.model_dynamics = model_dynamics
        self.stochastic_processes = self._get_stochastic_processes()
        self.stochastic_process_indices = self._get_stochastic_process_indices()
        self.num_trajectories = num_trajectories  # now safe to trigger the setter

        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory if initial_inventory is not None else np.zeros(num_assets)
        self.max_inventory = max_inventory
        if seed:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.start_time = start_time

        self.max_stock_price = max_stock_price or self.model_dynamics.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.model_dynamics.state = self.initial_state
        self.info_calculator = info_calculator
        self._empty_infos = self._get_empty_infos()

        self.observation_space = self._get_observation_space()
        self.action_space = self.model_dynamics.get_action_space()
        self.normalise_action_space_ = normalise_action_space
        self.normalise_observation_space_ = normalise_observation_space
        self.normalise_rewards_ = normalise_rewards

        if self.normalise_observation_space_:
            self.original_observation_space = copy(self.observation_space)
            self.observation_space = self._get_normalised_observation_space()
        if self.normalise_action_space_:
            self.original_action_space = copy(self.action_space)
            self.action_space = self._get_normalised_action_space()

    # By default normalised observations and actions are used. To display raw
    # observations after training, toggle the inverse flag in normalise_observation
    # and normalise_action.
    def reset(self):
        for process in self.stochastic_processes.values():
            # every stochastic process has a reset method
            process.reset()  # sets current_state to initial_state
        self.model_dynamics.state = self.initial_state  # initial_state is a property
        # the reward function also has a reset method; it takes initial_state as its argument
        self.reward_function.reset(self.model_dynamics.state.copy())
        return self.normalise_observation(self.model_dynamics.state.copy())

    # Steps the environment one step forward given the agent's action and returns
    # next_state, rewards, dones (per-trajectory flags) and infos.
    def step(self, action: np.ndarray):
        action = self.normalise_action(action, inverse=True)  # by default gets denormalised
        current_state = self.model_dynamics.state.copy()
        next_state = self._update_state(action)
        dones = self._get_dones()
        # All trajectories are synchronised (all done or not done together), so checking
        # the first trajectory's done status (dones[0]) is sufficient. The current state
        # is used to compute rewards before it is updated.
        rewards = self.reward_function.calculate(current_state, action, next_state, dones[0])
        infos = self._calculate_infos(current_state, action, rewards)
        return self.normalise_observation(next_state.copy()), self.normalise_rewards(rewards), dones, infos

    # Returns normalised observations by default (inverse=False).
    def normalise_observation(self, obs: np.ndarray, inverse: bool = False):
        if self.normalise_observation_space_ and not inverse:
            # _intercept_obs_norm is the lower bound; _gradient_obs_norm is the half-range
            # ((high - low) / 2). This scales the observation to (-1, 1).
            return (obs - self._intercept_obs_norm) / self._gradient_obs_norm - 1
        elif self.normalise_observation_space_ and inverse:
            return (obs + 1) * self._gradient_obs_norm + self._intercept_obs_norm  # denormalised obs
        else:
            return obs

    # Returns normalised actions by default.
    def normalise_action(self, action: np.ndarray, inverse: bool = False):
        if self.normalise_action_space_ and not inverse:
            return (action - self._intercept_action_norm) / self._gradient_action_norm - 1
        elif self.normalise_action_space_ and inverse:
            return (action + 1) * self._gradient_action_norm + self._intercept_action_norm
        else:
            return action

    def normalise_rewards(self, rewards: np.ndarray):
        if self.normalise_rewards_:
            scaled = rewards * self.reward_scaling
            return scaled
        return rewards

    # Returns the initial state of the environment.
    @property
    def initial_state(self):
        # Time -> shape (num_trajectories, 1)
        t = np.full((self.num_trajectories, 1), self.start_time)

        # Cash -> shape (num_trajectories, 1)
        cash = np.tile(np.array(self.initial_cash).reshape(1, 1), (self.num_trajectories, 1))

        # Inventory -> shape (num_trajectories, num_assets)
        inventory = self._get_initial_inventories()

        # Combine cash, inventory, time
        initial_state = np.concatenate([cash, inventory, t], axis=1)

        # Flatten and append stochastic process states
        for process in self.stochastic_processes.values():
            flat_vector = process.initial_vector_state.reshape(self.num_trajectories, -1)
            initial_state = np.concatenate([initial_state, flat_vector], axis=1)

        return initial_state

    # state is exposed as a property carrying the environment state.
    @property
    def state(self):
        return self.model_dynamics.state

    @property
    def is_at_max_inventory(self):
        idx = get_index_map(self.num_assets)
        return (self.state[:, idx["INVENTORY_INDEX"]:idx["TIME_INDEX"]] >= self.max_inventory).any(axis=1)

    @property
    def is_at_min_inventory(self):
        idx = get_index_map(self.num_assets)
        return (self.state[:, idx["INVENTORY_INDEX"]:idx["TIME_INDEX"]] <= -self.max_inventory).any(axis=1)

    # _step_size is for internal use; step_size is the public property (getter).
    @property
    def step_size(self):
        return self._step_size

    # Setter for step_size. Updating the property keeps _step_size and all dependent
    # components (stochastic processes, reward function) synchronised, so step_size can
    # be changed at runtime consistently.
    @step_size.setter
    def step_size(self, step_size: float):
        self._step_size = step_size
        for process_name, process in self.stochastic_processes.items():
            if process.step_size != step_size:  # each stochastic process has a step_size attribute
                process.step_size = step_size
        if hasattr(self.reward_function, "step_size"):  # not all reward functions use step_size
            self.reward_function.step_size = step_size

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @num_trajectories.setter
    def num_trajectories(self, num_trajectories: int):
        self._num_trajectories = num_trajectories
        for process_name, process in self.stochastic_processes.items():
            if process.num_trajectories != num_trajectories:
                process.num_trajectories = num_trajectories
        # If num_trajectories changes after initialization, refresh the dependent state.
        self._empty_infos = self._get_empty_infos()
        self.model_dynamics.fill_multiplier = self.model_dynamics._get_fill_multiplier()

    # Used for normalization
    @property
    def _intercept_obs_norm(self):
        return self.original_observation_space.low

    # Used for normalization
    @property
    def _gradient_obs_norm(self):
        return (self.original_observation_space.high - self.original_observation_space.low) / 2

    # Used for normalization
    @property
    def _intercept_action_norm(self):
        return self.original_action_space.low

    # Used for normalization
    @property
    def _gradient_action_norm(self):
        return (self.original_action_space.high - self.original_action_space.low) / 2

    # state[0]=cash, state[1]=inventory, state[2]=time, state[3]=asset_price, and the
    # remaining entries depend on the dimensionality of the arrival, midprice and fill
    # probability processes.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        arrivals, fills = self.model_dynamics.get_arrivals_and_fills(action)
        if fills is not None:
            fills = self._remove_max_inventory_fills(fills)  # adjusted fills
        # update cash, inventory and time from the adjusted fills, arrivals and actions
        self._update_agent_state(arrivals, fills, action)  # these methods update model_dynamics.state
        self._update_market_state(arrivals, fills, action)  # rather than returning anything
        return self.model_dynamics.state

    # Updates all processes and writes the updated values into model_dynamics.state.
    def _update_market_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        for process_name, process in self.stochastic_processes.items():
            process.update(arrivals, fills, action, self.model_dynamics.state)
            lower_index = self.stochastic_process_indices[process_name][0]
            upper_index = self.stochastic_process_indices[process_name][1]
            flat_state = process.current_state.reshape(self.num_trajectories, -1)  # flatten
            self.model_dynamics.state[:, lower_index:upper_index] = flat_state


    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.model_dynamics.update_state(arrivals, fills, action)  # inventory and cash get updated
        self._clip_inventory_and_cash()  # limits the range between min and max values
        self.model_dynamics.state[:, self.index_map["TIME_INDEX"]] += self.step_size

    def _get_dones(self):
        # step_size / 2 is a small offset to account for numerical imprecision
        idx = get_index_map(self.num_assets)
        done = self.model_dynamics.state[0, idx["TIME_INDEX"]] >= self.terminal_time - self.step_size / 2

        return np.full((self.num_trajectories,), done, dtype=bool)

    # If info_calculator is None, returns empty infos. Otherwise delegates to the
    # info_calculator (see info_calculators.py for how to build a custom one).
    def _calculate_infos(self, current_state, action, rewards):
        return (
            self.info_calculator.calculate(current_state, action, rewards)
            if self.info_calculator is not None
            else self._empty_infos
        )

    # updated for the multiple-asset case
    def _get_max_cash(self) -> float:
        max_prices = self.model_dynamics.midprice_model.max_value[:, 0]  # shape: (num_assets,)
        return self.n_steps * np.sum(max_prices)


    def _get_observation_space(self) -> gym.spaces.Space:
        """Observation space: [cash, inventory (per asset), time, ...stochastic processes...]"""

        # Cash - shape (1,)
        low = [-self.max_cash]
        high = [self.max_cash]

        # Inventory - shape (num_assets,)
        low += [-self.max_inventory] * self.num_assets
        high += [self.max_inventory] * self.num_assets

        # Time - shape (1,)
        low += [0.0]
        high += [self.terminal_time]

        # Append stochastic process bounds (flattened)
        for process in self.stochastic_processes.values():
            low.extend(process.min_value.flatten())
            high.extend(process.max_value.flatten())

        return Box(low=np.float32(low), high=np.float32(high))


    def _get_normalised_observation_space(self):
        # Linear normalisation of the gym.Box space so that the domain of the observation space is [-1,1].
        return gym.spaces.Box(
            low=-np.ones_like(self.observation_space.low, dtype=np.float32),
            high=np.ones_like(self.observation_space.high, dtype=np.float32),
        )

    def _get_normalised_action_space(self):
        # Linear normalisation of the gym.Box space so that the domain of the action space is [-1,1].
        return gym.spaces.Box(
            low=-np.ones_like(self.action_space.low, dtype=np.float32),
            high=np.ones_like(self.action_space.high, dtype=np.float32),
        )

    # Returns the discretised start time (defaults to 0 if start_time is not given).
    def _get_start_time(self):
        if isinstance(self.start_time, (float, int)):
            random_start = self.start_time
        elif isinstance(self.start_time, Callable):
            random_start = self.start_time()
        else:
            raise NotImplementedError
        return self._quantise_time_to_step(random_start)

    # Rounds start_time to a multiple of step_size.
    def _quantise_time_to_step(self, time: float):
        assert (time >= 0.0) and (time < self.terminal_time), "Start time is not within (0, env.terminal_time)."
        return np.round(time / self.step_size) * self.step_size

    # modified for multiple assets
    def _get_initial_inventories(self) -> np.ndarray:
        if isinstance(self.initial_inventory, tuple) and len(self.initial_inventory) == 2:
            # Randomize per asset per trajectory
            return self.rng.integers(
                *self.initial_inventory, size=(self.num_trajectories, self.model_dynamics.num_assets)
            )
        elif isinstance(self.initial_inventory, int):
            return np.full((self.num_trajectories, self.model_dynamics.num_assets), self.initial_inventory)
        elif isinstance(self.initial_inventory, np.ndarray):
            assert self.initial_inventory.shape == (self.model_dynamics.num_assets,)
            return np.tile(self.initial_inventory, (self.num_trajectories, 1))
        elif isinstance(self.initial_inventory, Callable):
            value = self.initial_inventory()
            if isinstance(value, int):
                return np.full((self.num_trajectories, self.model_dynamics.num_assets), value)
            else:
                return value
        else:
            raise Exception("Initial inventory must be an int, a tuple of length 2, a NumPy array, or a callable.")

    # Modified for the multi-asset case. Called from _update_agent_state (via _update_state,
    # itself called by step) to clip inventory and cash to their min/max ranges.
    def _clip_inventory_and_cash(self):
        idx = get_index_map(self.num_assets)
        # Clip multi-asset inventory block
        self.model_dynamics.state[:, idx["INVENTORY_INDEX"]:idx["TIME_INDEX"]] = self._clip(
            self.model_dynamics.state[:, idx["INVENTORY_INDEX"]:idx["TIME_INDEX"]],
            -self.max_inventory,
            self.max_inventory,
            cash_flag=False
        )
        # Clip cash (still scalar per trajectory)
        self.model_dynamics.state[:, idx["CASH_INDEX"]] = self._clip(
            self.model_dynamics.state[:, idx["CASH_INDEX"]],
            -self.max_cash,
            self.max_cash,
            cash_flag=True
        )


    # Used by _clip_inventory_and_cash; returns clipped cash/inventory and prints a
    # notice if any value was clipped.
    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped).any() and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped).any() and not cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    # Defined here but not currently called anywhere within Bek_TradingEnvironment.
    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)

    def _get_stochastic_processes(self):
        stochastic_processes = dict()
        for process_name in ["midprice_model", "arrival_model", "fill_probability_model", "price_impact_model", "lob_depth_model"]:
            process: StochasticProcessModel = getattr(self.model_dynamics, process_name)
            if process is not None:
                stochastic_processes[process_name] = process
        return OrderedDict(stochastic_processes)

    # Modified for multiple assets
    def _get_stochastic_process_indices(self):
        process_indices = dict()
        count = 2 + self.num_assets

        for process_name, process in self.stochastic_processes.items():
            flat = process.initial_vector_state.reshape(self.num_trajectories, -1)
            dimension = flat.shape[1]
            process_indices[process_name] = (count, count + dimension)
            count += dimension
        self._expected_total_state_dim = count  # save for use elsewhere
        return OrderedDict(process_indices)




    # Returns a list of empty dicts, one per trajectory, e.g. [{}, {}, {}, {}, {}].
    def _get_empty_infos(self):
        return [{} for _ in range(self.num_trajectories)] if self.num_trajectories > 1 else {}

    # Modified for multiple assets
    def _remove_max_inventory_fills(self, fills: np.ndarray) -> np.ndarray:
        # Boolean masks: shape (n_trajectories, 1)
        mask_max = (~self.is_at_max_inventory).reshape(-1, 1)  # True if not at max -> can buy
        mask_min = (~self.is_at_min_inventory).reshape(-1, 1)  # True if not at min -> can sell

        # Repeat masks for each asset
        mask_buy = np.repeat(mask_max, self.num_assets, axis=1)   # asset 0 bid, asset 1 bid, ...
        mask_sell = np.repeat(mask_min, self.num_assets, axis=1)  # asset 0 ask, asset 1 ask, ...

        fill_mask = np.stack([mask_buy, mask_sell], axis=2)  # shape (num_trajectories, num_assets, 2)
        return fill_mask * fills  # same shape: (num_trajectories, num_assets, 2)



    # Not currently used. Kept for future reward normalization or benchmarking.
    def _get_inventory_neutral_rewards(self, num_total_trajectories=100_000):
        fixed_action = 1
        full_trajectory_env = deepcopy(self)
        full_trajectory_env.start_time = 0.0
        full_trajectory_env.num_trajectories = num_total_trajectories
        full_trajectory_env.normalise_rewards_ = False

        class FixedAgent(Agent):
            def get_action(self, obs: np.ndarray) -> np.ndarray:
                return np.ones((num_total_trajectories, 2)) * fixed_action

        fixed_agent = FixedAgent()
        _, _, rewards = generate_trajectory(full_trajectory_env, fixed_agent)
        mean_rewards = np.mean(rewards) * self.n_steps
        return mean_rewards

    # Adding i + 1 gives each stochastic process a distinct seed even when the same seed
    # is passed to the environment, avoiding correlated randomness across processes. Use a
    # fixed seed for reproducible evaluation; avoid seeding the training environment to
    # encourage generalization. The magnitude of the seed value does not matter.
    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        for i, process in enumerate(self.stochastic_processes.values()):
            process.seed(seed + i + 1)
