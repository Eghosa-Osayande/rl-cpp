from __future__ import annotations

import math
from typing import Any

import numpy as np

from sgcnn.interfaces import (
    EpisodeSpec,
    MotionInterface,
    MotionStepResult,
    PoseState,
)


class GridMotion(MotionInterface):
    def __init__(
        self,
        *,
        motion_max_lin_vel: float = 0.26,
        motion_max_ang_vel: float = 1.0,
        motion_max_lin_acc: float | None = None,
        motion_max_ang_acc: float | None = None,
        motion_action_delay: float = 0.0,
        motion_position_noise: float = 0.01,
        motion_heading_noise: float = 0.05,
    ):
        self._max_lin_vel = float(motion_max_lin_vel)
        self._max_ang_vel = float(motion_max_ang_vel)
        self.max_lin_acc = motion_max_lin_acc
        self.max_ang_acc = motion_max_ang_acc
        self.action_delay = float(motion_action_delay)
        self.position_noise = float(motion_position_noise)
        self.heading_noise = float(motion_heading_noise)
        self._episode: EpisodeSpec | None = None
        self._position_m: np.ndarray | None = None
        self._heading: float | None = None
        self._linvel = 0.0
        self._angvel = 0.0
        self._prev_lin_cmd = 0.0
        self._prev_ang_cmd = 0.0

    @property
    def max_lin_vel(self) -> float:
        return self._max_lin_vel

    @property
    def max_ang_vel(self) -> float:
        return self._max_ang_vel

    def reset(self, episode: EpisodeSpec) -> PoseState:
        self._episode = episode
        if episode.spawn_heading is not None:
            heading = float(episode.spawn_heading)
        else:
            heading = float(np.random.uniform(0, 2 * math.pi))
        if episode.spawn_position_m is not None:
            position_m = np.array(episode.spawn_position_m, dtype=float)
        else:
            position_m = np.random.uniform(size=2) * float(episode.size_m)
        return self.set_pose(position_m, heading)

    def step_linear_angular(
        self,
        linear_velocity: float,
        angular_velocity: float,
        dt: float,
        **kwargs: Any,
    ) -> MotionStepResult:
        del kwargs
        if self._position_m is None or self._heading is None:
            raise RuntimeError("Motion must be reset before stepping.")
        linear_velocity = float(np.clip(linear_velocity, -self.max_lin_vel, self.max_lin_vel))
        angular_velocity = float(np.clip(angular_velocity, -self.max_ang_vel, self.max_ang_vel))
        position_m = self._position_m.copy()
        heading = self._heading

        if self.action_delay > 0 and dt > self.action_delay:
            position_m, heading = self._integrate(
                linear_velocity=self._prev_lin_cmd,
                angular_velocity=self._prev_ang_cmd,
                duration=self.action_delay,
                position_m=position_m,
                heading=heading,
            )
            position_m, heading = self._integrate(
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                duration=dt - self.action_delay,
                position_m=position_m,
                heading=heading,
            )
        else:
            position_m, heading = self._integrate(
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                duration=dt,
                position_m=position_m,
                heading=heading,
            )

        self._prev_lin_cmd = linear_velocity
        self._prev_ang_cmd = angular_velocity
        self._position_m = position_m
        self._heading = heading
        pose_state = self._make_pose_state(position_m, heading)
        return MotionStepResult(
            pose_state=pose_state,
            max_lin_vel=self.max_lin_vel,
            max_ang_vel=self.max_ang_vel,
            metadata={
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity,
            },
        )

    def step_wheel_velocities(
        self,
        left_velocity: float,
        right_velocity: float,
        dt: float,
        **kwargs: Any,
    ) -> MotionStepResult:
        del kwargs
        if self._episode is None:
            raise RuntimeError("Motion must be reset before stepping.")
        linear_velocity = 0.5 * (left_velocity + right_velocity)
        angular_velocity = (right_velocity - left_velocity) / self._episode.wheel_base
        return self.step_linear_angular(linear_velocity, angular_velocity, dt)

    def set_pose(self, position_m: np.ndarray, heading: float) -> PoseState:
        self._position_m = np.array(position_m, dtype=float)
        self._heading = float(heading) % (2 * math.pi)
        self._linvel = 0.0
        self._angvel = 0.0
        self._prev_lin_cmd = 0.0
        self._prev_ang_cmd = 0.0
        return self._make_pose_state(self._position_m, self._heading)

    def _integrate(
        self,
        *,
        linear_velocity: float,
        angular_velocity: float,
        duration: float,
        position_m: np.ndarray,
        heading: float,
    ) -> tuple[np.ndarray, float]:
        if self.max_lin_acc is None:
            self._linvel = linear_velocity
        else:
            diff = linear_velocity - self._linvel
            max_diff = self.max_lin_acc * duration
            if abs(diff) > max_diff:
                self._linvel += math.copysign(max_diff, diff)
            else:
                self._linvel = linear_velocity

        if self.max_ang_acc is None:
            self._angvel = angular_velocity
        else:
            diff = angular_velocity - self._angvel
            max_diff = self.max_ang_acc * duration
            if abs(diff) > max_diff:
                self._angvel += math.copysign(max_diff, diff)
            else:
                self._angvel = angular_velocity

        pos_diff = self._linvel * duration
        ang_diff = self._angvel * duration
        new_heading = (heading + ang_diff) % (2 * math.pi)
        inter_heading = heading + ang_diff / 2
        pos_vec = pos_diff * np.array([math.cos(inter_heading), math.sin(inter_heading)])
        return position_m + pos_vec, new_heading

    def _make_pose_state(self, position_m: np.ndarray, heading: float) -> PoseState:
        noisy_heading = float(np.random.normal(loc=heading, scale=self.heading_noise) % (2 * math.pi))
        noisy_position_m = np.random.normal(loc=position_m, scale=self.position_noise)
        return PoseState(
            position_m=np.array(position_m, dtype=float, copy=True),
            heading=float(heading),
            noisy_position_m=np.array(noisy_position_m, dtype=float, copy=True),
            noisy_heading=noisy_heading,
        )
