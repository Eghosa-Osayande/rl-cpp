import os
from typing import Callable
from stable_baselines3.common.callbacks import CheckpointCallback


class FnCheckpointCallback(CheckpointCallback):

    def __init__(
        self,
        save_freq_fn: Callable[[], int],
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer_fn: Callable[[], bool] = lambda: False,
        save_vecnormalize_fn: Callable[[], bool] = lambda: False,
        use_static_files: Callable[[], bool] = lambda: False,
        verbose: int = 0,
    ):
        self.use_static_files = use_static_files
        self.save_freq_fn = save_freq_fn
        self.save_vecnormalize_fn = save_vecnormalize_fn
        self.save_replay_buffer_fn = save_replay_buffer_fn
        super().__init__(
            save_freq=self.save_freq_fn(),
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=self.save_replay_buffer_fn(),
            save_vecnormalize=self.save_vecnormalize_fn(),
            verbose=verbose,
        )

    def _on_step(self) -> bool:
        self.save_freq = self.save_freq_fn()
        self.save_vecnormalize = self.save_vecnormalize_fn()
        self.save_replay_buffer = self.save_replay_buffer_fn()
        return super()._on_step()

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:

        if self.use_static_files() and checkpoint_type in {
            "replay_buffer_",
            "vecnormalize_",
        }:
            return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}_static.{extension}")

        return os.path.join(
            self.save_path,
            f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps:07d}_steps.{extension}",
        )
