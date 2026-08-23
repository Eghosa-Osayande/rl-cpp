from __future__ import annotations

import math
from typing import Any

import numpy as np

from sgcnn.interfaces import EpisodeSpec, PoseState, RangeState, RangingInterface


class GridLidarRanging(RangingInterface):
    def __init__(
        self,
        *,
        range_num_beams: int = 24,
        range_max_distance: float = 3.5,
        range_fov_degrees: float = 345.0,
        range_noise: float = 0.05,
    ):
        self._range_num_beams = int(range_num_beams)
        self.range_max_distance = float(range_max_distance)
        self.range_fov_degrees = float(range_fov_degrees)
        self.range_noise = float(range_noise)
        self._true_obstacle_map: np.ndarray | None = None
        self._pixels_per_meter = 1.0
        self._size_p = 0

    @property
    def range_num_beams(self) -> int:
        return self._range_num_beams

    def reset(
        self,
        episode: EpisodeSpec,
        pose_state: PoseState,
    ) -> RangeState | None:
        self._true_obstacle_map = np.maximum(
            episode.known_obstacle_map,
            episode.unknown_obstacle_map,
        )
        self._pixels_per_meter = 1.0 / episode.meters_per_pixel
        self._size_p = int(episode.size_p)
        return self.scan(pose_state)

    def scan(
        self,
        pose_state: PoseState,
        dt: float | None = None,
        **kwargs: Any,
    ) -> RangeState | None:
        del dt, kwargs
        if self._true_obstacle_map is None or self.range_num_beams <= 0:
            raise RuntimeError("Ranging must be reset before scanning.")
        distances = np.ones(self.range_num_beams, dtype=np.float32) * self.range_max_distance
        hits = np.zeros(self.range_num_beams, dtype=bool)
        samples = max(1, int(self.range_max_distance * self._pixels_per_meter))
        position_p = pose_state.position_m * self._pixels_per_meter
        angles_degrees = np.linspace(
            -self.range_fov_degrees / 2,
            self.range_fov_degrees / 2,
            num=self.range_num_beams,
            dtype=np.float32,
        )
        angles_radians = np.deg2rad(angles_degrees).astype(np.float32)
        for index, angle in enumerate(angles_radians):
            ang = pose_state.heading + float(angle)
            search_vec = np.array([math.cos(ang), math.sin(ang)])
            hit = None
            for sample_index in range(samples):
                pos_p = position_p + (sample_index + 1) * search_vec
                i = int(pos_p[0])
                j = int(pos_p[1])
                if i < 0 or i >= self._size_p or j < 0 or j >= self._size_p:
                    hit = np.array([i, j], dtype=np.int32)
                    break
                if self._true_obstacle_map[i, j] > 0:
                    hit = np.array([i, j], dtype=np.int32)
                    hits[index] = True
                    break
            if hit is None:
                pos_p = position_p + samples * search_vec
                hit = np.array([int(pos_p[0]), int(pos_p[1])], dtype=np.int32)
            offset_m = hit / self._pixels_per_meter - pose_state.position_m
            distance = math.sqrt(offset_m[0] ** 2 + offset_m[1] ** 2)
            distance = float(np.random.normal(loc=distance, scale=self.range_noise))
            distances[index] = np.clip(distance, 0, self.range_max_distance)
        sample_pairs = np.column_stack((distances, angles_radians)).astype(np.float32)
        return RangeState(
            samples=sample_pairs,
            max_distance=self.range_max_distance,
            metadata={
                "hits": hits,
                "angle_unit": "radians",
            },
        )
