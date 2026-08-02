

from __future__ import annotations

import os
from typing import Iterable, Optional

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class AccurateEvalCallback(EvalCallback):
    """
    EvalCallback with:
      - a counter for how many evals actually ran
      - optional reseeding of eval env right before evaluation
      - tracking the timestep when the "best model" was saved (heuristic)
    """

    def __init__(self, *args, seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_eval_calls = 0
        self.best_model_step = None
        self.seed = seed

    def _on_step(self) -> bool:
        # Eval triggers when n_calls % eval_freq == 0
        if (self.n_calls % self.eval_freq) == 0:
            self.actual_eval_calls += 1

            # Optional: force eval randomness to be repeatable
            if self.seed is not None:
                self.eval_env.seed(self.seed)

            # Run SB3 evaluation logic
            super()._on_step()

            # Heuristic: if best_mean_reward just matched last_mean_reward, assume best was saved
            if self.best_mean_reward == self.last_mean_reward:
                self.best_model_step = self.num_timesteps

        return True


class SaveAtTimestepsCallback(BaseCallback):
    """
    Saves model checkpoints at specified timesteps (milestones).
    Example: save_steps=[100_000, 200_000, 500_000, 1_000_000]
    """

    def __init__(
        self,
        save_steps: Iterable[int],
        save_dir: str,
        name_prefix: str = "PPO",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_steps = sorted(int(s) for s in save_steps)
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self._saved = set()
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def _format_step(step: int) -> str:
        if step % 1_000_000 == 0:
            return f"{step // 1_000_000}M"
        if step % 1_000 == 0:
            return f"{step // 1_000}K"
        return str(step)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        for s in self.save_steps:
            if s not in self._saved and t >= s:
                tag = self._format_step(s)
                path = os.path.join(self.save_dir, f"{self.name_prefix}_{tag}")
                self.model.save(path)
                self._saved.add(s)

                if self.verbose:
                    print(f"Saved checkpoint at {s:,} steps -> {path}")

        return True
