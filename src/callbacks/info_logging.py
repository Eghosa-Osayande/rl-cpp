from typing import Any
from stable_baselines3.common.callbacks import BaseCallback

from .buffered_csv import BufferedCSVCallback


class InfoLoggingCallback(BaseCallback):
    def __init__(
        self,
        filepath: str,
        verbose: int = 0,
        get_buffer_size=lambda: 100,
        info_key: str = None,
        log_keys: list[dict[str : list[str]]] = None,
    ):
        self.log_keys = log_keys if log_keys else []
        self.info_key = info_key
        super().__init__(verbose)
        self._initialized = False
        self.get_buffer_size = get_buffer_size

        self.csv_callback = BufferedCSVCallback(
            filepath=filepath,
            get_buffer_size=self.get_buffer_size,
        )

    def _initialize_from_row(self, row: dict[str, Any]) -> None:
        if self._initialized:
            return

        self.csv_callback.set_fieldnames(list(row.keys()))
        self._initialized = True

    def _process_row(self, row: dict[str, Any]) -> None:
        self._initialize_from_row(row)
        self.csv_callback.add_row(row)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            if (
                isinstance(info, dict)
                and self.info_key
                and self.info_key in info
                and isinstance(info[self.info_key], dict)
            ):
                logs = info["logs"]
                self._process_row(logs)

                if env_idx < len(dones) and bool(dones[env_idx]):
                    for item in self.log_keys:
                        prefix=item["prefix"]
                        title=item["title"]
                            
                        for field in item["fields"]:
                            if field not in logs:
                                continue
                            logger_key = f"{title}/{field.removeprefix(prefix)}"
                            self.logger.record(logger_key, logs[field])

        return True

    def _on_training_end(self) -> None:
        self.csv_callback.flush()
