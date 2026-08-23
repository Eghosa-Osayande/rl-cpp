from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from sgcnn.interfaces import EpisodeResult, EpisodeSpec


def normalize_ranges(range_distances: np.ndarray, max_distance: float) -> np.ndarray:
    return np.clip(range_distances / max_distance, 0, 1).astype(np.float32)


def make_history_buffer(initial_value: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("History buffer length must be positive.")
    return np.repeat(initial_value[np.newaxis, ...], length, axis=0).astype(np.float32)


def write_history(buffer: np.ndarray, write_index: int, value: np.ndarray) -> None:
    buffer[write_index % buffer.shape[0]] = value


def read_history(buffer: np.ndarray, start_index: int) -> np.ndarray:
    n = start_index % buffer.shape[0]
    return np.concatenate((buffer[n:], buffer[:n]), axis=0)


def collision_reward(
    collision_type: str | None,
    reward_collision_wall: float,
    reward_collision_obstacle: float,
) -> float:
    if collision_type == "wall":
        return reward_collision_wall
    if collision_type == "obstacle":
        return reward_collision_obstacle
    return 0.0


def angle_diff(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b)
    return min(diff, 2 * math.pi - diff)


def maybe_final_map_image(
    known_obstacle_map: np.ndarray,
    unknown_obstacle_map: np.ndarray,
    done: bool,
) -> np.ndarray | None:
    if not done:
        return None
    img = 255 * np.ones_like(known_obstacle_map, dtype=np.uint8)
    img[known_obstacle_map > 0] = 0
    img[unknown_obstacle_map > 0] = 0
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def make_episode_result(
    episode: EpisodeSpec,
    reached_goal: bool,
    elapsed_steps: int,
    coverage_in_percent: float,
    truncated: bool,
    metrics: dict[str, Any] | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        episode=episode,
        reached_goal=reached_goal,
        elapsed_steps=elapsed_steps,
        coverage_in_percent=coverage_in_percent,
        truncated=truncated,
        metrics=metrics,
    )
