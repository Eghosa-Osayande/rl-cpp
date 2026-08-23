from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseState:
    position_m: np.ndarray
    heading: float
    noisy_position_m: np.ndarray | None = None
    noisy_heading: float | None = None


@dataclass(frozen=True)
class MotionStepResult:
    pose_state: PoseState
    max_lin_vel: float
    max_ang_vel: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RangeState:
    samples: np.ndarray
    max_distance: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CollisionStepResult:
    pose_state: PoseState
    collided: bool
    collision_type: str | None = None
    last_safe_pose_state: PoseState | None = None
    collision_index: int | None = None


class MotionInterface(ABC):
    @property
    @abstractmethod
    def max_lin_vel(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_ang_vel(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self, episode: EpisodeSpec) -> PoseState:
        raise NotImplementedError

    @abstractmethod
    def step_linear_angular(
        self,
        linear_velocity: float,
        angular_velocity: float,
        dt: float,
        **kwargs: Any,
    ) -> MotionStepResult:
        raise NotImplementedError

    @abstractmethod
    def step_wheel_velocities(
        self,
        left_velocity: float,
        right_velocity: float,
        dt: float,
        **kwargs: Any,
    ) -> MotionStepResult:
        raise NotImplementedError

    @abstractmethod
    def set_pose(self, position_m: np.ndarray, heading: float) -> PoseState:
        raise NotImplementedError


class RangingInterface(ABC):
    @property
    @abstractmethod
    def range_num_beams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        episode: EpisodeSpec,
        pose_state: PoseState,
    ) -> RangeState | None:
        raise NotImplementedError

    @abstractmethod
    def scan(
        self,
        pose_state: PoseState,
        dt: float | None = None,
        **kwargs: Any,
    ) -> RangeState | None:
        raise NotImplementedError


class CollisionInterface(ABC):
    @abstractmethod
    def reset(self, episode: EpisodeSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def resolve_path(
        self,
        path_pose_states: list[PoseState],
        **kwargs: Any,
    ) -> CollisionStepResult:
        raise NotImplementedError


@dataclass(frozen=True)
class EpisodeSpec:
    meters_per_pixel: float
    size_p: int
    size_m: float
    mower_radius: float
    wheel_base: float
    known_obstacle_map: np.ndarray
    unknown_obstacle_map: np.ndarray
    goal_coverage: float
    goal_steps: int
    body_width: float | None = None
    body_length: float | None = None
    all_unknown: bool = False
    filename: str | None = None
    current_map: int | None = None
    use_floor_plans: bool = False
    use_known_obstacles: bool = False
    use_unknown_obstacles: bool = False
    eval_mode: bool = False
    spawn_position_m: np.ndarray | None = None
    spawn_heading: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeResult:
    episode: EpisodeSpec
    reached_goal: bool
    elapsed_steps: int
    coverage_in_percent: float
    truncated: bool
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class CurriculumProgress:
    level: int
    completed_maps: int
    total_maps: int
    completed_floor_plans: bool
    completed_obstacles: bool


class CurriculumManager(ABC):
    @abstractmethod
    def next_episode(self) -> EpisodeSpec:
        raise NotImplementedError

    @abstractmethod
    def on_episode_end(self, result: EpisodeResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_progress(self) -> CurriculumProgress:
        raise NotImplementedError
