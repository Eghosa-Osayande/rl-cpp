from __future__ import annotations
from dataclasses import dataclass

import math
from typing import Any

import cv2
import numpy as np

from .interfaces import (
    EpisodeSpec,
    PoseState,
    RangeState,
)


@dataclass(frozen=True)
class MapperState:
    coverage_map: np.ndarray
    overlap_map: np.ndarray
    known_obstacle_map: np.ndarray
    unknown_obstacle_map: np.ndarray
    frontier_map: np.ndarray | None
    coverage_in_pixels: float
    coverage_in_m2: float
    coverage_in_percent: float
    overlap_in_pixels: float
    overlap_in_m2: float
    overlap_in_percent: float
    global_total_variation: float
    frontier_distance: float
    coverage_state_projected: np.ndarray
    obstacle_state_projected: np.ndarray
    frontier_state_projected: np.ndarray | None = None


@dataclass(frozen=True)
class MapStepState:
    mapper_state: MapperState
    newly_visited_sum: int
    newly_visited_dilated_sum: int
    local_coverage_diff: np.ndarray
    local_coverage_diff_dilated_sum: float
    total_variation_diff: float
    frontier_distance_before: float
    frontier_distance_after: float
    local_overlap_before: np.ndarray


class RangeCoverageMapper:
    def __init__(
        self,
        *,
        mapper_exploration: bool = False,
        mapper_overlap_observation: bool = True,
        mapper_frontier_observation: bool = True,
        mapper_obstacle_dilation: int = 9,
        mapper_input_size: int = 32,
        mapper_num_maps: int = 4,
        mapper_scale_factor: int = 4,
        mapper_coverage_pad_value: int = 0,
        mapper_obstacle_pad_value: int = 1,
        mapper_use_known_obstacles_in_tv: bool = True,
        mapper_use_unknown_obstacles_in_tv: bool = True,
    ):
        self.exploration = bool(mapper_exploration)
        self.overlap_observation = bool(mapper_overlap_observation)
        self.frontier_observation = bool(mapper_frontier_observation)
        self.obstacle_dilation = int(mapper_obstacle_dilation)
        self.input_size = int(mapper_input_size)
        self.num_maps = int(mapper_num_maps)
        self.scale_factor = int(mapper_scale_factor)
        self.base_coverage_pad_value = int(mapper_coverage_pad_value)
        self.base_obstacle_pad_value = int(mapper_obstacle_pad_value)
        self.use_known_obstacles_in_tv = bool(mapper_use_known_obstacles_in_tv)
        self.use_unknown_obstacles_in_tv = bool(mapper_use_unknown_obstacles_in_tv)
        self.ms_reach_p = self.input_size * self.scale_factor ** (self.num_maps - 1)
        self._episode: EpisodeSpec | None = None
        self._pose_state: PoseState | None = None
        self._range_state: RangeState | None = None
        self._lidar_pts: np.ndarray | None = None
        self._lidar_pts_max: np.ndarray | None = None
        self._old_lidar_pts: np.ndarray | None = None
        self._old_lidar_pts_max: np.ndarray | None = None
        self._head_vec_old: np.ndarray | None = None
        self._orth_vec_old: np.ndarray | None = None
        self._mapper_state: MapperState | None = None

    def reset(
        self,
        episode: EpisodeSpec,
        pose_state: PoseState,
        range_state: RangeState | None,
    ) -> MapperState:
        self._episode = episode
        self._pose_state = pose_state
        self._range_state = range_state
        self.meters_per_pixel = float(episode.meters_per_pixel)
        self.pixels_per_meter = 1.0 / self.meters_per_pixel
        self.size_p = int(episode.size_p)
        self.size_m = float(episode.size_m)
        self.coverage_pad_value = self.base_coverage_pad_value
        self.obstacle_pad_value = self.base_obstacle_pad_value
        self.known_obstacle_map = np.array(
            episode.known_obstacle_map, dtype=float, copy=True
        )
        self.unknown_obstacle_map = np.array(
            episode.unknown_obstacle_map, dtype=float, copy=True
        )
        if episode.all_unknown:
            self.coverage_pad_value = 0
            self.obstacle_pad_value = 0

        self._lidar_pts, self._lidar_pts_max = self._compute_range_geometry(
            pose_state, range_state
        )
        self._old_lidar_pts = None
        self._old_lidar_pts_max = None
        self._head_vec_old = None
        self._orth_vec_old = None
        self._update_obstacle_maps_from_range_state(range_state)
        self.coverage_map = np.zeros((self.size_p, self.size_p), dtype=float)
        mower_radius = episode.mower_radius
        if not self.exploration:
            cv2.circle(
                self.coverage_map,
                center=np.flip(pose_state.position_m * self.pixels_per_meter).astype(
                    np.int32
                ),
                radius=int(mower_radius * self.pixels_per_meter),
                color=1,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_8,
            )
        else:
            if self._lidar_pts is not None and len(self._lidar_pts) > 0:
                polygon_points = self._prepare_polygon_points(
                    self._lidar_pts,
                    anchor_point=pose_state.position_m * self.pixels_per_meter,
                )
                if len(polygon_points) >= 3:
                    cv2.fillPoly(
                        self.coverage_map,
                        [polygon_points],
                        color=1,
                    )
            idxs = np.logical_or(self.known_obstacle_map, self.unknown_obstacle_map)
            self.coverage_map[idxs] = 0
        self.overlap_map = self.coverage_map.copy()
        self.frontier_map = self._compute_frontier_map(
            self.coverage_map, self.known_obstacle_map
        )
        self.global_total_variation, self.frontier_distance = (
            self._initialize_state_metrics(pose_state)
        )
        self._mapper_state = self._build_mapper_state(pose_state)
        return self._mapper_state

    def step(
        self,
        previous_pose_state: PoseState,
        pose_state: PoseState,
        range_state: RangeState | None,
    ) -> MapStepState:
        if self._mapper_state is None or self._episode is None:
            raise RuntimeError("Mapper must be reset before stepping.")
        self._old_lidar_pts = (
            None if self._lidar_pts is None else self._lidar_pts.copy()
        )
        self._old_lidar_pts_max = (
            None if self._lidar_pts_max is None else self._lidar_pts_max.copy()
        )
        self._pose_state = pose_state
        self._range_state = range_state
        self._lidar_pts, self._lidar_pts_max = self._compute_range_geometry(
            pose_state, range_state
        )
        mower_radius = self._episode.mower_radius
        radius_m = (
            mower_radius
            if not self.exploration
            else max(mower_radius, self._range_max_distance(range_state))
        )
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(
            previous_pose_state.position_m,
            pose_state.position_m,
            radius_m,
        )
        local_overlap_before = self.overlap_map[i1:i2, j1:j2].copy()
        local_coverage_before = self.coverage_map[i1:i2, j1:j2].copy()
        local_known_before = self.known_obstacle_map[i1:i2, j1:j2].copy()
        local_unknown_before = self.unknown_obstacle_map[i1:i2, j1:j2].copy()
        self._update_obstacle_maps_from_range_state(range_state)

        local_coverage_diff = self._compute_coverage_diff(
            previous_pose_state=previous_pose_state,
            pose_state=pose_state,
            i1=i1,
            i2=i2,
            j1=j1,
            j2=j2,
            local_coverage_before=local_coverage_before,
        )
        self.overlap_map[i1:i2, j1:j2] += local_coverage_diff
        self.coverage_map[i1:i2, j1:j2] = self.overlap_map[i1:i2, j1:j2].clip(max=1)

        d = max(1, 2 + self.obstacle_dilation // 2)
        ii1 = max(0, i1 - d)
        ii2 = min(self.size_p, i2 + d)
        jj1 = max(0, j1 - d)
        jj2 = min(self.size_p, j2 + d)
        local_frontier = self._compute_frontier_map(
            self.coverage_map[ii1:ii2, jj1:jj2],
            self.known_obstacle_map[ii1:ii2, jj1:jj2],
        )
        self.frontier_map[i1:i2, j1:j2] = local_frontier[
            i1 - ii1 : i2 - ii1,
            j1 - jj1 : j2 - jj1,
        ]

        local_coverage_after = self.coverage_map[i1:i2, j1:j2]
        newly_visited = local_coverage_after > local_coverage_before
        newly_visited_sum = int(newly_visited.sum())
        all_obstacles = np.maximum(
            self.known_obstacle_map[i1:i2, j1:j2],
            self.unknown_obstacle_map[i1:i2, j1:j2],
        )
        if self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,) * 2, dtype=float)
            all_obstacles = self._dilate_obstacles(all_obstacles, kernel)
        newly_visited_dilated = newly_visited.copy()
        newly_visited_dilated[all_obstacles > 0] = 0
        newly_visited_dilated_sum = int(newly_visited_dilated.sum())
        local_coverage_diff_dilated = local_coverage_diff.copy()
        local_coverage_diff_dilated[all_obstacles > 0] = 0
        local_coverage_diff_dilated_sum = float(local_coverage_diff_dilated.sum())

        (
            coverage_in_pixels,
            coverage_in_m2,
            coverage_in_percent,
            overlap_in_pixels,
            overlap_in_m2,
            overlap_in_percent,
        ) = self._update_global_metrics(
            newly_visited_dilated_sum=newly_visited_dilated_sum,
            local_coverage_diff_dilated_sum=local_coverage_diff_dilated_sum,
        )

        local_obstacles_old = None
        local_obstacles_new = None
        if self.use_known_obstacles_in_tv and self.use_unknown_obstacles_in_tv:
            local_obstacles_old = np.maximum(local_known_before, local_unknown_before)
            local_obstacles_new = np.maximum(
                self.known_obstacle_map[i1:i2, j1:j2],
                self.unknown_obstacle_map[i1:i2, j1:j2],
            )
        elif self.use_known_obstacles_in_tv:
            local_obstacles_old = local_known_before
            local_obstacles_new = self.known_obstacle_map[i1:i2, j1:j2].copy()
        elif self.use_unknown_obstacles_in_tv:
            local_obstacles_old = local_unknown_before
            local_obstacles_new = self.unknown_obstacle_map[i1:i2, j1:j2].copy()

        if local_obstacles_new is not None and self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,) * 2, dtype=float)
            local_obstacles_old = self._dilate_obstacles(local_obstacles_old, kernel)
            local_obstacles_new = self._dilate_obstacles(local_obstacles_new, kernel)
        total_variation_old = total_variation(
            local_coverage_before, local_obstacles_old
        )
        total_variation_new = total_variation(local_coverage_after, local_obstacles_new)
        total_variation_diff = total_variation_new - total_variation_old
        frontier_distance_before = self.frontier_distance
        self.global_total_variation += total_variation_diff
        self.coverage_in_pixels = coverage_in_pixels
        self.coverage_in_m2 = coverage_in_m2
        self.coverage_in_percent = coverage_in_percent
        self.overlap_in_pixels = overlap_in_pixels
        self.overlap_in_m2 = overlap_in_m2
        self.overlap_in_percent = overlap_in_percent
        self.frontier_distance = self._compute_frontier_distance(pose_state)
        self._mapper_state = self._build_mapper_state(pose_state)
        return MapStepState(
            mapper_state=self._mapper_state,
            newly_visited_sum=newly_visited_sum,
            newly_visited_dilated_sum=newly_visited_dilated_sum,
            local_coverage_diff=local_coverage_diff.astype(np.float32),
            local_coverage_diff_dilated_sum=local_coverage_diff_dilated_sum,
            total_variation_diff=total_variation_diff,
            frontier_distance_before=frontier_distance_before,
            frontier_distance_after=self.frontier_distance,
            local_overlap_before=local_overlap_before.astype(np.float32),
        )

    def render(self) -> np.ndarray | None:
        if (
            self._pose_state is None
            or self._mapper_state is None
            or self._episode is None
        ):
            return None
        img = np.ones((self.size_p, self.size_p, 3), dtype=float)
        overlap_tanh = np.tanh(0.5 * self.coverage_map)
        img[:, :, 0] = np.clip(1 - 1.5 * overlap_tanh, 0, 1)
        img[:, :, 1] = np.clip(2 - 1.5 * overlap_tanh, 0, 1)
        img[:, :, 2] = np.clip(1 - 1.5 * overlap_tanh, 0, 1)
        img[self.unknown_obstacle_map > 0] = 0.5
        img[self.known_obstacle_map > 0] = 0
        if self.frontier_observation and self.frontier_map is not None:
            img[self.frontier_map > 0, 0] = 1
            img[self.frontier_map > 0, 1] = 0
            img[self.frontier_map > 0, 2] = 1
        cv2.circle(
            img,
            center=np.flip(self._pose_state.position_m * self.pixels_per_meter).astype(
                np.int32
            ),
            radius=int(self._episode.mower_radius * self.pixels_per_meter),
            color=[1, 0, 0],
            thickness=cv2.FILLED,
            lineType=cv2.LINE_8,
        )
        if self._lidar_pts is not None:
            for pt in self._lidar_pts:
                cv2.line(
                    img,
                    np.flip(self._pose_state.position_m * self.pixels_per_meter).astype(
                        np.int32
                    ),
                    np.flip(pt).astype(np.int32),
                    color=(0, 0, 1),
                    thickness=1,
                    lineType=cv2.LINE_8,
                )
        return (255 * np.flip(img.transpose(1, 0, 2), axis=0)).astype(np.uint8)

    def get_state(self) -> MapperState:
        if self._mapper_state is None:
            raise RuntimeError("Mapper state is not initialized.")
        return self._mapper_state

    def _build_mapper_state(self, pose_state: PoseState) -> MapperState:
        cov_map = (
            np.tanh(0.2 * self.overlap_map)
            if self.overlap_observation
            else self.coverage_map
        )
        coverage_state_projected = self._get_multi_scale_map(
            pose_state, cov_map, self.coverage_pad_value
        )
        obstacle_state_projected = self._get_multi_scale_map(
            pose_state,
            self.known_obstacle_map,
            self.obstacle_pad_value,
        )
        frontier_state_projected = None
        if self.frontier_observation and self.frontier_map is not None:
            frontier_state_projected = self._get_multi_scale_map(
                pose_state,
                self.frontier_map,
                0,
                ceil=True,
            )
        return MapperState(
            coverage_map=self.coverage_map.copy(),
            overlap_map=self.overlap_map.copy(),
            known_obstacle_map=self.known_obstacle_map.copy(),
            unknown_obstacle_map=self.unknown_obstacle_map.copy(),
            frontier_map=(
                None if self.frontier_map is None else self.frontier_map.copy()
            ),
            coverage_in_pixels=self.coverage_in_pixels,
            coverage_in_m2=self.coverage_in_m2,
            coverage_in_percent=self.coverage_in_percent,
            overlap_in_pixels=self.overlap_in_pixels,
            overlap_in_m2=self.overlap_in_m2,
            overlap_in_percent=self.overlap_in_percent,
            global_total_variation=self.global_total_variation,
            frontier_distance=self.frontier_distance,
            coverage_state_projected=coverage_state_projected.astype(np.float32),
            obstacle_state_projected=obstacle_state_projected.astype(np.float32),
            frontier_state_projected=(
                None
                if frontier_state_projected is None
                else frontier_state_projected.astype(np.float32)
            ),
        )

    def _initialize_state_metrics(self, pose_state: PoseState) -> tuple[float, float]:
        all_obstacles = np.maximum(self.known_obstacle_map, self.unknown_obstacle_map)
        if self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,) * 2, dtype=float)
            all_obstacles = self._dilate_obstacles(all_obstacles, kernel)
        self.total_obstacle_pixels_dilated = float(all_obstacles.sum())
        self.free_space_dilated = self.size_p**2 - self.total_obstacle_pixels_dilated

        coverage = self.coverage_map.copy()
        coverage[all_obstacles > 0] = 0
        overlap = self.overlap_map.copy()
        overlap[all_obstacles > 0] = 0
        self.coverage_in_pixels = float(coverage.sum())
        self.coverage_in_m2 = self.coverage_in_pixels / (self.pixels_per_meter**2)
        self.coverage_in_percent = self.coverage_in_pixels / self.free_space_dilated
        self.overlap_in_pixels = float((overlap - coverage).sum())
        self.overlap_in_m2 = self.overlap_in_pixels / (self.pixels_per_meter**2)
        self.overlap_in_percent = (
            self.overlap_in_pixels / self.coverage_in_pixels
            if self.coverage_in_pixels
            else 0.0
        )
        mower_radius = self._episode.mower_radius
        radius_m = (
            mower_radius
            if not self.exploration
            else max(mower_radius, self._range_max_distance(self._range_state))
        )
        local_coverage = self._get_local_neighborhood(
            self.coverage_map,
            pose_state.position_m,
            pose_state.position_m,
            radius_m,
        )
        local_obstacles = None
        if self.use_known_obstacles_in_tv and self.use_unknown_obstacles_in_tv:
            local_known = self._get_local_neighborhood(
                self.known_obstacle_map,
                pose_state.position_m,
                pose_state.position_m,
                radius_m,
            )
            local_unknown = self._get_local_neighborhood(
                self.unknown_obstacle_map,
                pose_state.position_m,
                pose_state.position_m,
                radius_m,
            )
            local_obstacles = np.maximum(local_known, local_unknown)
        elif self.use_known_obstacles_in_tv:
            local_obstacles = self._get_local_neighborhood(
                self.known_obstacle_map,
                pose_state.position_m,
                pose_state.position_m,
                radius_m,
            ).copy()
        elif self.use_unknown_obstacles_in_tv:
            local_obstacles = self._get_local_neighborhood(
                self.unknown_obstacle_map,
                pose_state.position_m,
                pose_state.position_m,
                radius_m,
            ).copy()
        if local_obstacles is not None and self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,) * 2, dtype=float)
            local_obstacles = self._dilate_obstacles(local_obstacles, kernel)
        global_total_variation = total_variation(local_coverage, local_obstacles)
        frontier_distance = self._compute_frontier_distance(pose_state)
        return global_total_variation, frontier_distance

    def _update_global_metrics(
        self,
        *,
        newly_visited_dilated_sum: int,
        local_coverage_diff_dilated_sum: float,
    ) -> tuple[float, float, float, float, float, float]:
        coverage_in_pixels = self.coverage_in_pixels + newly_visited_dilated_sum
        coverage_in_m2 = coverage_in_pixels / (self.pixels_per_meter**2)
        coverage_in_percent = coverage_in_pixels / self.free_space_dilated
        overlap_in_pixels = self.overlap_in_pixels + (
            local_coverage_diff_dilated_sum - newly_visited_dilated_sum
        )
        overlap_in_m2 = overlap_in_pixels / (self.pixels_per_meter**2)
        overlap_in_percent = (
            overlap_in_pixels / coverage_in_pixels if coverage_in_pixels else 0.0
        )
        return (
            coverage_in_pixels,
            coverage_in_m2,
            coverage_in_percent,
            overlap_in_pixels,
            overlap_in_m2,
            overlap_in_percent,
        )

    def _compute_range_geometry(
        self,
        pose_state: PoseState,
        range_state: RangeState | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if range_state is None:
            return None, None
        samples = np.asarray(range_state.samples, dtype=float)
        if samples.size == 0:
            return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

        distances = samples[:, 0]
        relative_angles = samples[:, 1]
        metadata = range_state.metadata or {}
        angle_unit = metadata.get("angle_unit", "radians")
        if angle_unit == "degrees":
            relative_angles = np.deg2rad(relative_angles)
        absolute_angles = pose_state.heading + relative_angles
        position_p = pose_state.position_m * self.pixels_per_meter
        search_vectors = np.column_stack(
            (np.cos(absolute_angles), np.sin(absolute_angles))
        )
        lidar_pts = (
            position_p
            + distances[:, np.newaxis] * self.pixels_per_meter * search_vectors
        )
        max_distance = float(range_state.max_distance)
        lidar_pts_max = (
            position_p + max_distance * self.pixels_per_meter * search_vectors
        )

        if len(relative_angles) > 1:
            angle_step = float(np.median(np.diff(relative_angles)))
        else:
            angle_step = 2 * math.pi

        extra_before: list[np.ndarray] = []
        angle = float(relative_angles[0] - angle_step)
        while angle > -math.pi:
            search_vec = np.array(
                [
                    math.cos(pose_state.heading + angle),
                    math.sin(pose_state.heading + angle),
                ]
            )
            extra_before.append(
                position_p + max_distance * self.pixels_per_meter * search_vec
            )
            angle -= angle_step

        extra_after: list[np.ndarray] = []
        angle = float(relative_angles[-1] + angle_step)
        while angle < math.pi:
            search_vec = np.array(
                [
                    math.cos(pose_state.heading + angle),
                    math.sin(pose_state.heading + angle),
                ]
            )
            extra_after.append(
                position_p + max_distance * self.pixels_per_meter * search_vec
            )
            angle += angle_step

        if extra_before or extra_after:
            pieces = []
            if extra_before:
                pieces.append(np.array(extra_before[::-1], dtype=float))
            pieces.append(lidar_pts_max)
            if extra_after:
                pieces.append(np.array(extra_after, dtype=float))
            lidar_pts_max = np.concatenate(pieces, axis=0)

        return lidar_pts.astype(np.int32), lidar_pts_max.astype(np.int32)

    def _update_obstacle_maps_from_range_state(
        self, range_state: RangeState | None
    ) -> None:
        if range_state is None or self._lidar_pts is None:
            return
        metadata = range_state.metadata or {}
        hits = metadata.get("hits")
        if hits is None:
            return
        for hit, pt in zip(hits, self._lidar_pts):
            if not hit:
                continue
            i = int(pt[0])
            j = int(pt[1])
            if (
                0 <= i < self.size_p
                and 0 <= j < self.size_p
                and self.unknown_obstacle_map[i, j] > 0
            ):
                self.unknown_obstacle_map[i, j] = 0
                self.known_obstacle_map[i, j] = 1

    def _compute_coverage_diff(
        self,
        *,
        previous_pose_state: PoseState,
        pose_state: PoseState,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        local_coverage_before: np.ndarray,
    ) -> np.ndarray:
        local_coverage_diff = np.zeros((i2 - i1, j2 - j1), dtype=float)
        mower_radius_p = self._episode.mower_radius * self.pixels_per_meter
        if not self.exploration:
            position_p = pose_state.position_m * self.pixels_per_meter
            previous_position_p = previous_pose_state.position_m * self.pixels_per_meter
            cv2.circle(
                local_coverage_diff,
                center=np.flip(position_p - [i1, j1]).astype(np.int32),
                radius=int(mower_radius_p),
                color=1,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_8,
            )
            head_vec_new = position_p - previous_position_p
            head_vec_len = float(np.linalg.norm(head_vec_new))
            orth_vec_new = None
            if head_vec_len > 1e-5:
                orth_vec_new = np.array([-head_vec_new[1], head_vec_new[0]])
                orth_vec_new *= mower_radius_p / head_vec_len
                cv2.fillConvexPoly(
                    local_coverage_diff,
                    points=np.array(
                        [
                            np.flip(previous_position_p - [i1, j1] + orth_vec_new),
                            np.flip(position_p - [i1, j1] + orth_vec_new),
                            np.flip(position_p - [i1, j1] - orth_vec_new),
                            np.flip(previous_position_p - [i1, j1] - orth_vec_new),
                        ]
                    ).astype(np.int32),
                    color=1,
                    lineType=cv2.LINE_8,
                )
            cv2.circle(
                local_coverage_diff,
                center=np.flip(previous_position_p - [i1, j1]).astype(np.int32),
                radius=int(mower_radius_p),
                color=0,
                thickness=cv2.FILLED,
                lineType=cv2.LINE_8,
            )
            if self._orth_vec_old is not None and self._head_vec_old is not None:
                cv2.fillConvexPoly(
                    local_coverage_diff,
                    points=np.array(
                        [
                            np.flip(
                                previous_position_p - [i1, j1] + self._orth_vec_old
                            ),
                            np.flip(
                                previous_position_p
                                - [i1, j1]
                                + self._orth_vec_old
                                - self._head_vec_old
                            ),
                            np.flip(
                                previous_position_p
                                - [i1, j1]
                                - self._orth_vec_old
                                - self._head_vec_old
                            ),
                            np.flip(
                                previous_position_p - [i1, j1] - self._orth_vec_old
                            ),
                        ]
                    ).astype(np.int32),
                    color=0,
                    lineType=cv2.LINE_8,
                )
            self._head_vec_old = head_vec_new
            self._orth_vec_old = orth_vec_new
        else:
            if (
                self._lidar_pts is not None
                and self._old_lidar_pts is not None
                and len(self._lidar_pts) > 0
            ):
                position_p = pose_state.position_m * self.pixels_per_meter
                previous_position_p = (
                    previous_pose_state.position_m * self.pixels_per_meter
                )
                current_polygon = self._prepare_polygon_points(
                    self._lidar_pts,
                    anchor_point=position_p,
                    offset=np.array([i1, j1], dtype=float),
                )
                if len(current_polygon) >= 3:
                    cv2.fillPoly(
                        local_coverage_diff,
                        [current_polygon],
                        color=1,
                    )
                previous_polygon = self._prepare_polygon_points(
                    self._old_lidar_pts,
                    anchor_point=previous_position_p,
                    offset=np.array([i1, j1], dtype=float),
                )
                if len(previous_polygon) >= 3:
                    cv2.fillPoly(
                        local_coverage_diff,
                        [previous_polygon],
                        color=0,
                    )
                overlappable_pixels = np.ones_like(local_coverage_diff)
                if (
                    self._old_lidar_pts_max is not None
                    and len(self._old_lidar_pts_max) > 0
                ):
                    overlap_polygon = self._prepare_polygon_points(
                        self._old_lidar_pts_max,
                        offset=np.array([i1, j1], dtype=float),
                    )
                    if len(overlap_polygon) >= 3:
                        cv2.fillPoly(
                            overlappable_pixels,
                            [overlap_polygon],
                            color=0,
                        )
                overlappable_pixels[local_coverage_before == 0] = 1
                local_coverage_diff *= overlappable_pixels
        idxs = np.logical_or(
            self.known_obstacle_map[i1:i2, j1:j2],
            self.unknown_obstacle_map[i1:i2, j1:j2],
        )
        local_coverage_diff[idxs] = 0
        return local_coverage_diff

    def _compute_frontier_map(
        self, coverage_map: np.ndarray, obstacle_map: np.ndarray
    ) -> np.ndarray:
        coverage_map = coverage_map.copy()
        obstacle_map = obstacle_map.copy()
        if self.obstacle_dilation > 1:
            kernel = np.ones((self.obstacle_dilation,) * 2, dtype=float)
            obstacle_map = self._dilate_obstacles(obstacle_map, kernel)
        coverage_map[obstacle_map > 0] = 0
        free_map = np.logical_not(coverage_map + obstacle_map)
        kernel = np.ones((3, 3), dtype=float)
        coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)
        return np.logical_and(coverage_map, free_map).astype(float)

    def _compute_frontier_distance(self, pose_state: PoseState) -> float:
        if not self.frontier_observation or self.frontier_map is None:
            return 0.0
        ms_frontier_map = self._get_multi_scale_map(
            pose_state, self.frontier_map, 0, ceil=True
        )
        center = (self.input_size - 1) / 2
        xx, yy = np.meshgrid(range(self.input_size), range(self.input_size))
        dist_map = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        for index in range(self.num_maps):
            distances = dist_map[ms_frontier_map[index] > 0]
            if len(distances) > 0:
                min_distance = float(min(distances))
                min_distance *= self.meters_per_pixel
                min_distance *= self.scale_factor**index
                return min_distance
        min_distance = self.input_size / np.sqrt(2)
        min_distance *= self.meters_per_pixel
        min_distance *= self.scale_factor ** (self.num_maps - 1)
        return float(min_distance)

    def _prepare_polygon_points(
        self,
        points: np.ndarray,
        *,
        anchor_point: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> np.ndarray:
        polygon_points = np.asarray(points, dtype=float).reshape(-1, 2)
        if anchor_point is not None:
            anchor = np.asarray(anchor_point, dtype=float).reshape(1, 2)
            polygon_points = np.concatenate((polygon_points, anchor), axis=0)
        if offset is not None:
            polygon_points = polygon_points - np.asarray(offset, dtype=float)
        return np.fliplr(polygon_points).astype(np.int32)

    def _dilate_obstacles(
        self, obstacle_map: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        dilated = obstacle_map.copy()
        dilated[0, :] = 1
        dilated[-1, :] = 1
        dilated[:, 0] = 1
        dilated[:, -1] = 1
        return cv2.dilate(dilated, kernel, iterations=1)

    def _get_transform_matrix(self, pose_state: PoseState, scale: float) -> np.ndarray:
        noisy_heading = (
            pose_state.noisy_heading
            if pose_state.noisy_heading is not None
            else pose_state.heading
        )
        noisy_position_m = (
            pose_state.noisy_position_m
            if pose_state.noisy_position_m is not None
            else pose_state.position_m
        )
        noisy_position_p = noisy_position_m * self.pixels_per_meter
        heading_degrees = noisy_heading * 180 / math.pi
        translation_matrix_1 = np.eye(3)
        translation_matrix_1[0, 2] = -noisy_position_p[1] / scale
        translation_matrix_1[1, 2] = -noisy_position_p[0] / scale
        rotation_matrix = np.eye(3)
        rotation_matrix[:2] = cv2.getRotationMatrix2D(
            center=(0, 0), angle=90 - heading_degrees, scale=1
        )
        translation_matrix_2 = np.eye(3)
        translation_matrix_2[0, 2] = self.input_size / 2
        translation_matrix_2[1, 2] = self.input_size / 2
        return translation_matrix_2 @ rotation_matrix @ translation_matrix_1

    def _get_relative_map(
        self,
        pose_state: PoseState,
        map_data: np.ndarray,
        pad_value: float,
        scale: float = 1,
        ceil: bool = False,
        floor: bool = False,
    ) -> np.ndarray:
        assert not (ceil and floor)
        sc = min(scale, self.size_p)
        matrix = self._get_transform_matrix(pose_state, sc)
        relative_map = cv2.resize(
            map_data,
            (int(0.5 + self.size_p / sc),) * 2,
            interpolation=cv2.INTER_AREA,
        )
        relative_map = cv2.warpAffine(
            relative_map,
            M=matrix[:2],
            dsize=(self.input_size,) * 2,
            borderValue=pad_value,
            flags=cv2.INTER_AREA,
        )
        if ceil:
            relative_map = np.ceil(relative_map)
        if floor:
            relative_map = np.floor(relative_map)
        return relative_map

    def _get_multi_scale_map(
        self,
        pose_state: PoseState,
        map_data: np.ndarray,
        pad_value: float,
        ceil: bool = False,
        floor: bool = False,
    ) -> np.ndarray:
        return np.array(
            [
                self._get_relative_map(
                    pose_state,
                    map_data,
                    pad_value,
                    scale=self.scale_factor**index,
                    ceil=ceil,
                    floor=floor,
                )
                for index in range(self.num_maps)
            ],
            dtype=np.float32,
        )

    def _get_local_neighborhood_indices(
        self,
        pos1_m: np.ndarray,
        pos2_m: np.ndarray,
        radius_m: float,
    ) -> tuple[int, int, int, int]:
        i1 = min(pos1_m[0], pos2_m[0]) - radius_m
        i2 = max(pos1_m[0], pos2_m[0]) + radius_m
        j1 = min(pos1_m[1], pos2_m[1]) - radius_m
        j2 = max(pos1_m[1], pos2_m[1]) + radius_m
        i1 = max(0, min(self.size_p, int(i1 * self.pixels_per_meter - 10)))
        i2 = max(0, min(self.size_p, int(i2 * self.pixels_per_meter + 10)))
        j1 = max(0, min(self.size_p, int(j1 * self.pixels_per_meter - 10)))
        j2 = max(0, min(self.size_p, int(j2 * self.pixels_per_meter + 10)))
        return i1, i2, j1, j2

    def _get_local_neighborhood(
        self,
        map_data: np.ndarray,
        pos1_m: np.ndarray,
        pos2_m: np.ndarray,
        radius_m: float,
    ) -> np.ndarray:
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(pos1_m, pos2_m, radius_m)
        return map_data[i1:i2, j1:j2]

    def _range_max_distance(self, range_state: RangeState | None) -> float:
        if range_state is None:
            return 0.0
        return float(range_state.max_distance)


def total_variation(
    img: np.ndarray,
    img2: np.ndarray | None = None,
    mode: str = "sym-iso",
) -> float:
    assert mode in ["sym-iso", "non-sym-iso", "non-iso"]
    img = img.astype(float)
    diff1 = np.abs(img[1:, :] - img[:-1, :])
    diff2 = np.abs(img[:, 1:] - img[:, :-1])
    if img2 is not None:
        assert img.shape == img2.shape
        img2 = np.maximum(img, img2)
        diff1 = np.minimum(diff1, np.abs(img2[1:, :] - img2[:-1, :]))
        diff2 = np.minimum(diff2, np.abs(img2[:, 1:] - img2[:, :-1]))
    if mode == "sym-iso":
        tv = np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[1:, :] ** 2))
        tv += np.sum(np.sqrt(diff1[:, 1:] ** 2 + diff2[:-1, :] ** 2))
        tv += np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2))
        tv += np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[1:, :] ** 2))
        return float(tv / 4)
    if mode == "non-sym-iso":
        return float(np.sum(np.sqrt(diff1[:, :-1] ** 2 + diff2[:-1, :] ** 2)))
    return float(np.sum(diff1) + np.sum(diff2))
