import os
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback


class StopperCallback(BaseCallback):

    def __init__(
        self,
        verbose: int = 0,
        stop_training_fn: Callable[[], bool] = lambda: False,
    ):
        self.stop_training_fn = stop_training_fn

        super().__init__(
            verbose=verbose,
        )

    def _on_step(self) -> bool:
        return not self.stop_training_fn()
