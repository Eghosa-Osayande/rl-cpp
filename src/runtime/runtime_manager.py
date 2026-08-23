import json
import threading
import time
import copy
from dataclasses import dataclass, asdict
from typing import Any, Callable


@dataclass
class RuntimeConfig:
    render_enabled: bool = False
    csv_buffer_size: int = 100
    refresh_frequency: int = 1
    checkpoint_save_freq: int = 1000
    checkpoint_save_replay_buffer: bool = False
    checkpoint_save_vecnormalize: bool = False
    end_training: bool = False
    checkpoint_use_static_files:bool=False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> "RuntimeConfig":
        if isinstance(data, str):
            data = json.loads(data)

        return cls(**data)


class RuntimeManager:

    def __init__(
        self,
        read_fn=None,
    ):
        self._lock = threading.Lock()
        self._config = RuntimeConfig()
        self._buffer = []
        self._is_working = False

        self.experiment_dir = ""
        self.renders_dir = ""
        self.info_file = ""

        self._worker = threading.Thread(
            target=self.config_reader_thread,
            daemon=True,
            kwargs={
                "read_fn": read_fn,
                "initial_config": self._config,
                "buffer": self._buffer,
                "lock": self._lock,
            },
        )

    def start(self):

        if self._is_working:
            return

        self._worker.start()
        self._is_working = True

    def get_config(self):
        if self._lock.acquire(blocking=False):
            try:
                if len(self._buffer) > 0:
                    self._config = self._buffer[-1]
            finally:
                self._lock.release()

        return self._config

    def set_config(self, new_config):
        new_config = copy.deepcopy(new_config)
        if self._lock.acquire(blocking=True):
            try:
                if len(self._buffer) > 0:
                    self._buffer[-1] = new_config
                else:
                    self._buffer.append(new_config)
                self._config = new_config
            finally:
                self._lock.release()

    def set_directories(
        self,
        experiment_dir="",
        renders_dir="",
        info_file="",
    ):
        self.experiment_dir = experiment_dir
        self.renders_dir = renders_dir
        self.info_file = info_file

    @property
    def can_render(self):
        return self.get_config().render_enabled

    @property
    def csv_buffer_size(self):
        return self.get_config().csv_buffer_size

    @property
    def checkpoint_save_freq(self):
        return self.get_config().checkpoint_save_freq

    @property
    def checkpoint_save_replay_buffer(self):
        return self.get_config().checkpoint_save_replay_buffer

    @property
    def checkpoint_save_vecnormalize(self):
        return self.get_config().checkpoint_save_vecnormalize
    
    @property
    def checkpoint_use_static_files(self):
        return self.get_config().checkpoint_use_static_files

    @property
    def end_training(self):
        return self.get_config().end_training

    @staticmethod
    def config_reader_thread(
        read_fn: Callable[[], RuntimeConfig],
        initial_config: RuntimeConfig,
        buffer: list,
        lock: threading.Lock,
    ):
        config = initial_config
        while not config.end_training and read_fn:
            try:
                if read_fn is not None:
                    config = read_fn()
                    with lock:
                        if len(buffer) > 0:
                            buffer[-1] = config
                        else:
                            buffer.append(config)
            except:
                pass

            time.sleep(config.refresh_frequency)


def read_config_from_file(path: str):
    with open(path) as f:
        new_data = json.load(f)
        return RuntimeConfig(
            render_enabled=new_data["render_enabled"],
            refresh_frequency=new_data["refresh_freq"],
            csv_buffer_size=new_data["csv_buffer_size"],
            checkpoint_save_freq=new_data["checkpoint_save_freq"],
            checkpoint_save_vecnormalize=new_data["checkpoint_save_vecnormalize"],
            checkpoint_save_replay_buffer=new_data["checkpoint_save_replay_buffer"],
            end_training=new_data["end_training"],
            checkpoint_use_static_files=new_data["checkpoint_use_static_files"],
        )


def read_config_url(path: str) -> RuntimeConfig:
    import requests
    response = requests.get(
        path,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    return RuntimeConfig.from_json(response.json())