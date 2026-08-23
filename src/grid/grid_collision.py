from __future__ import annotations

import cv2
import numpy as np

from sgcnn.interfaces import (
    CollisionInterface,
    CollisionStepResult,
    EpisodeSpec,
    PoseState,
)


class GridCollision(CollisionInterface):
    def __init__(
        self,
        *,
        collision_pose_policy: str = "revert_pose",
    ):
        self.pose_policy = collision_pose_policy
        self._episode: EpisodeSpec | None = None
        self._true_obstacle_map: np.ndarray | None = None
        self._pixels_per_meter = 1.0
        self._size_p = 0
        self._size_m = 0.0
        assert self.pose_policy in {
            "revert_pose",
            "keep_collided_pose",
            "snap_to_world",
        }

    def reset(self, episode: EpisodeSpec) -> None:
        self._episode = episode
        self._true_obstacle_map = np.maximum(
            episode.known_obstacle_map,
            episode.unknown_obstacle_map,
        )
        self._pixels_per_meter = 1.0 / episode.meters_per_pixel
        self._size_p = int(episode.size_p)
        self._size_m = float(episode.size_m)

    def resolve_path(
        self,
        path_pose_states: list[PoseState],
        **kwargs,
    ) -> CollisionStepResult:
        del kwargs
        if self._true_obstacle_map is None or self._episode is None:
            raise RuntimeError("Collision interface must be reset before use.")
        if not path_pose_states:
            raise ValueError("Path must contain at least one pose state.")

        start_pose_state = path_pose_states[0]
        if len(path_pose_states) == 1:
            collision_type = self._classify_collision(start_pose_state.position_m)
            return CollisionStepResult(
                pose_state=start_pose_state,
                collided=collision_type is not None,
                collision_type=collision_type,
            )

        last_safe_pose_state = start_pose_state
        for index, pose_state in enumerate(path_pose_states[1:], start=1):
            collision_type = self._classify_collision(pose_state.position_m)
            if collision_type is None:
                last_safe_pose_state = pose_state
                continue
            resolved_pose_state = self._resolve_collision_pose(
                start_pose_state=start_pose_state,
                last_safe_pose_state=last_safe_pose_state,
                colliding_pose_state=pose_state,
                is_segment=len(path_pose_states) == 2,
            )
            return CollisionStepResult(
                pose_state=resolved_pose_state,
                collided=True,
                collision_type=collision_type,
                last_safe_pose_state=last_safe_pose_state,
                collision_index=index,
            )
        return CollisionStepResult(
            pose_state=path_pose_states[-1],
            collided=False,
            collision_type=None,
            last_safe_pose_state=last_safe_pose_state,
            collision_index=None,
        )

    def _resolve_collision_pose(
        self,
        *,
        start_pose_state: PoseState,
        last_safe_pose_state: PoseState,
        colliding_pose_state: PoseState,
        is_segment: bool,
    ) -> PoseState:
        if self.pose_policy == "revert_pose":
            return start_pose_state
        if self.pose_policy == "keep_collided_pose":
            return colliding_pose_state
        if is_segment:
            snapped_position = self._snap_to_segment(
                previous_position_m=last_safe_pose_state.position_m,
                position_m=colliding_pose_state.position_m,
            )
            return PoseState(
                position_m=snapped_position,
                heading=colliding_pose_state.heading,
            )
        return last_safe_pose_state

    def _classify_collision(self, position_m: np.ndarray) -> str | None:
        if self._is_wall_collision(position_m):
            return "wall"
        if self._is_obstacle_collision(position_m):
            return "obstacle"
        return None

    def _is_wall_collision(self, position_m: np.ndarray) -> bool:
        radius_m = self._episode.mower_radius
        return (
            position_m[0] <= radius_m
            or position_m[1] <= radius_m
            or position_m[0] >= self._size_m - radius_m
            or position_m[1] >= self._size_m - radius_m
        )

    def _is_obstacle_collision(self, position_m: np.ndarray) -> bool:
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(position_m, position_m)
        local_obstacle_map = self._true_obstacle_map[i1:i2, j1:j2]
        local_position_map = np.zeros_like(local_obstacle_map)
        position_p = position_m * self._pixels_per_meter - [i1, j1]
        cv2.circle(
            local_position_map,
            center=np.flip(position_p).astype(np.int32),
            radius=int(self._episode.mower_radius * self._pixels_per_meter),
            color=1,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_8,
        )
        return bool(np.logical_and(local_position_map, local_obstacle_map).any())

    def _snap_to_segment(
        self,
        *,
        previous_position_m: np.ndarray,
        position_m: np.ndarray,
    ) -> np.ndarray:
        if self._classify_collision(position_m) is None:
            return np.array(position_m, dtype=float, copy=True)
        if self._classify_collision(previous_position_m) is not None:
            return np.array(previous_position_m, dtype=float, copy=True)
        low = 0.0
        high = 1.0
        direction = position_m - previous_position_m
        for _ in range(32):
            mid = (low + high) / 2
            candidate = previous_position_m + mid * direction
            if self._classify_collision(candidate) is not None:
                high = mid
            else:
                low = mid
        snapped = previous_position_m + low * direction
        if self._classify_collision(snapped) is not None:
            snapped = previous_position_m.copy()
        return snapped

    def _get_local_neighborhood_indices(
        self,
        pos1_m: np.ndarray,
        pos2_m: np.ndarray,
    ) -> tuple[int, int, int, int]:
        radius_m = self._episode.mower_radius
        i1 = min(pos1_m[0], pos2_m[0]) - radius_m
        i2 = max(pos1_m[0], pos2_m[0]) + radius_m
        j1 = min(pos1_m[1], pos2_m[1]) - radius_m
        j2 = max(pos1_m[1], pos2_m[1]) + radius_m
        i1 = max(0, min(self._size_p, int(i1 * self._pixels_per_meter - 10)))
        i2 = max(0, min(self._size_p, int(i2 * self._pixels_per_meter + 10)))
        j1 = max(0, min(self._size_p, int(j1 * self._pixels_per_meter - 10)))
        j2 = max(0, min(self._size_p, int(j2 * self._pixels_per_meter + 10)))
        return i1, i2, j1, j2
