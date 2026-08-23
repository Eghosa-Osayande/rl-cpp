from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np

from sgcnn.interfaces import (
    CurriculumManager,
    CurriculumProgress,
    EpisodeResult,
    EpisodeSpec,
)


class GridCurriculumManager(CurriculumManager):
    def __init__(
        self,
        meters_per_pixel: float = 0.0375,
        min_size: float | None = None,
        max_size: float | None = None,
        exploration: bool = False,
        eval: bool = False,
        all_unknown: bool = True,
        p_use_known_obstacles: float = 0.7,
        p_use_unknown_obstacles: float = 0.7,
        p_use_floor_plans: float = 0.7,
        max_known_obstacles: int = 100,
        max_unknown_obstacles: int = 100,
        obstacle_radius: float = 0.25,
        mower_radius: float = 0.15,
        start_level: int = 1,
        
    ):
        self.meters_per_pixel = meters_per_pixel
        self.pixels_per_meter = 1 / self.meters_per_pixel
        self.exploration = exploration
        self.eval = eval
        self.all_unknown = all_unknown
        self.p_use_known_obstacles = p_use_known_obstacles
        self.p_use_unknown_obstacles = p_use_unknown_obstacles
        self.p_use_floor_plans = p_use_floor_plans
        self.max_known_obstacles = max_known_obstacles
        self.max_unknown_obstacles = max_unknown_obstacles
        self.obstacle_radius = obstacle_radius
        self.mower_radius = mower_radius
        self.line_type = cv2.LINE_8
        if min_size is not None and max_size is not None:
            assert min_size <= max_size
        if min_size is not None:
            self.base_min_size_p = int(min_size * self.pixels_per_meter)
        elif self.exploration:
            self.base_min_size_p = 256
        else:
            self.base_min_size_p = 64
        if max_size is not None:
            self.base_max_size_p = int(max_size * self.pixels_per_meter)
            self.base_min_size_p = min(self.base_min_size_p, self.base_max_size_p)
        elif self.exploration:
            self.base_max_size_p = max(400, self.base_min_size_p)
        else:
            self.base_max_size_p = max(200, self.base_min_size_p)
        self.min_size_p = self.base_min_size_p
        self.max_size_p = self.base_max_size_p

        maps_dir = os.path.join(os.path.dirname(__file__), "maps")
        self.maps_dir = maps_dir
        if self.exploration:
            self.eval_maps = glob.glob(os.path.join(self.maps_dir, "eval_exploration*"))
        else:
            self.eval_maps = glob.glob(os.path.join(self.maps_dir, "eval_mowing*"))
        self.train_maps_0 = glob.glob(os.path.join(self.maps_dir, "train_0_*"))
        self.train_maps_1 = glob.glob(os.path.join(self.maps_dir, "train_1_*"))
        self.train_maps_2 = glob.glob(os.path.join(self.maps_dir, "train_2_*"))
        self.train_maps_3 = glob.glob(os.path.join(self.maps_dir, "train_3_*"))
        self.train_maps_4 = glob.glob(os.path.join(self.maps_dir, "train_4_*"))
        self.train_maps_5 = glob.glob(os.path.join(self.maps_dir, "train_5_*"))

        self.current_episode = 0
        self.level = start_level
        self.next_train_map = 0
        self.use_randomized_envs = False
        self.train_maps: list[str] = []
        self.goal_steps = 2000
        self.goal_coverage = 0.9
        self.requires_floor_plan_completion = False
        self.requires_obstacle_completion = False
        self.completed_maps: list[bool] = []
        self.completed_floor_plan = True
        self.completed_obstacles = True
        self._set_level(self.level, reset_progress=True)

    def next_episode(self) -> EpisodeSpec:
        self.current_episode += 1
        if self.eval:
            filename = self.eval_maps[(self.current_episode - 1) % len(self.eval_maps)]
            spec = self._build_map_episode(filename=filename, current_map=None)
        else:
            spec = self._create_training_episode()
        if self.all_unknown:
            spec = self._with_all_unknown(spec)
        return spec

    def on_episode_end(self, result: EpisodeResult) -> None:
        if result.episode.eval_mode or not result.reached_goal:
            return
        if result.elapsed_steps > result.episode.goal_steps:
            return
        episode = result.episode
        if episode.current_map is not None and episode.current_map < len(
            self.completed_maps
        ):
            self.completed_maps[episode.current_map] = True
        else:
            if episode.use_floor_plans:
                self.completed_floor_plan = True
            if episode.use_known_obstacles or episode.use_unknown_obstacles:
                self.completed_obstacles = True
        if (
            np.all(self.completed_maps)
            and self.completed_floor_plan
            and self.completed_obstacles
        ):
            self.level += 1
            self._set_level(self.level, reset_progress=True)

    def get_progress(self) -> CurriculumProgress:
        return CurriculumProgress(
            level=self.level,
            completed_maps=sum(self.completed_maps),
            total_maps=len(self.completed_maps),
            completed_floor_plans=self.completed_floor_plan,
            completed_obstacles=self.completed_obstacles,
        )

    def _set_level(self, level: int, reset_progress: bool) -> None:
        self.level = level
        self.min_size_p = self.base_min_size_p
        self.max_size_p = self.base_max_size_p
        self.goal_steps = 2000
        if self.exploration:
            if level == 1:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = self.train_maps_1 + self.train_maps_2
            elif level == 2:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_1 + self.train_maps_2 + self.train_maps_4
                )
            elif level == 3:
                self.goal_coverage = 0.95
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_1 + self.train_maps_2 + self.train_maps_4
                )
            elif level == 4:
                self.goal_coverage = 0.97
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_1 + self.train_maps_2 + self.train_maps_4
                )
            elif level == 5:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_1 + self.train_maps_2 + self.train_maps_4
                )
            elif level == 6:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_1
                    + self.train_maps_2
                    + self.train_maps_3
                    + self.train_maps_4
                )
            elif level == 7:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.min_size_p = 256
                self.max_size_p = 320
                self.train_maps = (
                    self.train_maps_1
                    + self.train_maps_2
                    + self.train_maps_3
                    + self.train_maps_4
                )
            elif level == 8:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.min_size_p = 256
                self.max_size_p = 400
                self.train_maps = (
                    self.train_maps_1
                    + self.train_maps_2
                    + self.train_maps_3
                    + self.train_maps_4
                )
            else:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.min_size_p = 256
                self.max_size_p = 400
                self.train_maps = (
                    self.train_maps_1
                    + self.train_maps_2
                    + self.train_maps_3
                    + self.train_maps_4
                    + self.train_maps_5
                )
        else:
            if level == 1:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = self.train_maps_0
            elif level == 2:
                self.goal_coverage = 0.9
                self.use_randomized_envs = False
                self.train_maps = self.train_maps_0 + self.train_maps_1
            elif level == 3:
                self.goal_coverage = 0.95
                self.use_randomized_envs = False
                self.train_maps = self.train_maps_0 + self.train_maps_1
            elif level == 4:
                self.goal_coverage = 0.95
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_0 + self.train_maps_1 + self.train_maps_2
                )
            elif level == 5:
                self.goal_coverage = 0.97
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_0 + self.train_maps_1 + self.train_maps_2
                )
            elif level == 6:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_0 + self.train_maps_1 + self.train_maps_2
                )
            elif level == 7:
                self.goal_coverage = 0.99
                self.use_randomized_envs = False
                self.train_maps = (
                    self.train_maps_0
                    + self.train_maps_1
                    + self.train_maps_2
                    + self.train_maps_3
                )
            else:
                self.goal_coverage = 0.99
                self.use_randomized_envs = True
                self.train_maps = (
                    self.train_maps_0
                    + self.train_maps_1
                    + self.train_maps_2
                    + self.train_maps_3
                )
        self.requires_floor_plan_completion = (
            self.use_randomized_envs and self.p_use_floor_plans > 0
        )
        use_known_obstacles = (
            self.max_known_obstacles > 0 and self.p_use_known_obstacles > 0
        )
        use_unknown_obstacles = (
            self.max_unknown_obstacles > 0 and self.p_use_unknown_obstacles > 0
        )
        self.requires_obstacle_completion = self.use_randomized_envs and (
            use_known_obstacles or use_unknown_obstacles
        )
        if reset_progress:
            self.completed_maps = [False] * len(self.train_maps)
            self.completed_floor_plan = not self.requires_floor_plan_completion
            self.completed_obstacles = not self.requires_obstacle_completion

    def _with_all_unknown(self, spec: EpisodeSpec) -> EpisodeSpec:
        unknown_obstacle_map = np.maximum(
            spec.known_obstacle_map, spec.unknown_obstacle_map
        ).copy()
        unknown_obstacle_map[0, :] = 1
        unknown_obstacle_map[-1, :] = 1
        unknown_obstacle_map[:, 0] = 1
        unknown_obstacle_map[:, -1] = 1
        return EpisodeSpec(
            meters_per_pixel=spec.meters_per_pixel,
            size_p=spec.size_p,
            size_m=spec.size_m,
            known_obstacle_map=np.zeros_like(spec.known_obstacle_map),
            unknown_obstacle_map=unknown_obstacle_map,
            goal_coverage=spec.goal_coverage,
            goal_steps=spec.goal_steps,
            all_unknown=True,
            filename=spec.filename,
            current_map=spec.current_map,
            use_floor_plans=spec.use_floor_plans,
            use_known_obstacles=spec.use_known_obstacles,
            use_unknown_obstacles=spec.use_unknown_obstacles,
            eval_mode=spec.eval_mode,
            spawn_position_m=spec.spawn_position_m,
            spawn_heading=spec.spawn_heading,
            mower_radius=self.mower_radius,
            wheel_base=self.mower_radius
        )

    def _build_map_episode(self, filename: str, current_map: int | None) -> EpisodeSpec:
        img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
        img = np.fliplr(img.transpose(1, 0))
        assert len(img.shape) == 2
        assert img.shape[0] == img.shape[1]
        assert set(img.flatten()).issubset([0, 128, 255]), "Invalid map image"
        size_p = img.shape[0]
        size_m = size_p / self.pixels_per_meter
        known_obstacle_map = np.zeros((size_p, size_p), dtype=float)
        unknown_obstacle_map = np.zeros((size_p, size_p), dtype=float)
        known_obstacle_map[img == 0] = 1
        unknown_obstacle_map[img == 128] = 1
        return EpisodeSpec(
            meters_per_pixel=self.meters_per_pixel,
            size_p=size_p,
            size_m=size_m,
            known_obstacle_map=known_obstacle_map,
            unknown_obstacle_map=unknown_obstacle_map,
            goal_coverage=self.goal_coverage,
            goal_steps=self.goal_steps,
            filename=filename,
            current_map=current_map,
            eval_mode=self.eval,
            mower_radius=self.mower_radius,
            wheel_base=self.mower_radius
        )

    def _create_training_episode(self) -> EpisodeSpec:
        current_map = None
        use_floor_plans = False
        use_known_obstacles = False
        use_unknown_obstacles = False
        filename = None
        if not self.use_randomized_envs or random.uniform(0, 1) < 0.5:
            current_map = self.next_train_map
            self.next_train_map = (self.next_train_map + 1) % len(self.train_maps)
            filename = self.train_maps[current_map]
            return self._build_map_episode(filename=filename, current_map=current_map)

        size_p = random.randint(self.min_size_p, self.max_size_p)
        size_m = size_p / self.pixels_per_meter
        known_obstacle_map = np.zeros((size_p, size_p), dtype=float)
        unknown_obstacle_map = np.zeros((size_p, size_p), dtype=float)
        use_floor_plans = random.uniform(0, 1) < self.p_use_floor_plans
        if use_floor_plans:
            self._randomize_floor_plan(known_obstacle_map, size_p)
        use_known_obstacles = (
            self.max_known_obstacles > 0
            and random.uniform(0, 1) < self.p_use_known_obstacles
        )
        use_unknown_obstacles = (
            self.max_unknown_obstacles > 0
            and random.uniform(0, 1) < self.p_use_unknown_obstacles
        )
        if use_known_obstacles or use_unknown_obstacles:
            self._randomize_circular_obstacles(
                known_obstacle_map=known_obstacle_map,
                unknown_obstacle_map=unknown_obstacle_map,
                size_p=size_p,
                size_m=size_m,
                use_known_obstacles=use_known_obstacles,
                use_unknown_obstacles=use_unknown_obstacles,
            )
        return EpisodeSpec(
            meters_per_pixel=self.meters_per_pixel,
            size_p=size_p,
            size_m=size_m,
            known_obstacle_map=known_obstacle_map,
            unknown_obstacle_map=unknown_obstacle_map,
            goal_coverage=self.goal_coverage,
            goal_steps=self.goal_steps,
            filename=filename,
            current_map=current_map,
            use_floor_plans=use_floor_plans,
            use_known_obstacles=use_known_obstacles,
            use_unknown_obstacles=use_unknown_obstacles,
            eval_mode=False,
            mower_radius=self.mower_radius,
            wheel_base=self.mower_radius
        )

    def _randomize_floor_plan(
        self, known_obstacle_map: np.ndarray, size_p: int
    ) -> None:
        min_room_size_p = int(10 * self.mower_radius * self.pixels_per_meter)
        max_room_size_p = int(32 * self.mower_radius * self.pixels_per_meter)
        min_wall_thickness_p = 2
        max_wall_thickness_p = int(2 * self.mower_radius * self.pixels_per_meter)
        min_gap = int(4 * self.mower_radius * self.pixels_per_meter)
        max_gap = int(8 * self.mower_radius * self.pixels_per_meter)
        if size_p <= 2 * min_room_size_p:
            return
        room_size_p = random.randint(min_room_size_p, max_room_size_p)
        num_walls = max(1, int(size_p / room_size_p) - 1)
        room_size_p = int(size_p / (num_walls + 1))
        wall_thickness_p = random.randint(min_wall_thickness_p, max_wall_thickness_p)
        vertical_stop = random.uniform(0, 1) < 0.5
        for n in range(num_walls):
            i1 = room_size_p * (n + 1) - wall_thickness_p // 2
            i2 = room_size_p * (n + 1) + wall_thickness_p
            if random.uniform(0, 1) < 0.9:
                known_obstacle_map[i1:i2, :] = 1
            if random.uniform(0, 1) < 0.9:
                known_obstacle_map[:, i1:i2] = 1
            stop_placed = False
            for m in range(num_walls + 1):
                g_min = min_gap
                g_max = min(max_gap, room_size_p - 2 * wall_thickness_p)
                j_min = room_size_p * m + wall_thickness_p
                j_max = room_size_p * (m + 1) - wall_thickness_p
                p_stop = 1 / (num_walls + 1 - m)
                place_stop = False
                if not stop_placed and random.uniform(0, 1) < p_stop:
                    place_stop = True
                    stop_placed = True
                if not vertical_stop or not place_stop:
                    gap = random.randint(g_min, g_max)
                    j1 = random.randint(j_min, j_max - gap)
                    j2 = j1 + gap
                    known_obstacle_map[i1:i2, j1:j2] = 0
                if vertical_stop or not place_stop:
                    gap = random.randint(g_min, g_max)
                    j1 = random.randint(j_min, j_max - gap)
                    j2 = j1 + gap
                    known_obstacle_map[j1:j2, i1:i2] = 0

    def _randomize_circular_obstacles(
        self,
        known_obstacle_map: np.ndarray,
        unknown_obstacle_map: np.ndarray,
        size_p: int,
        size_m: float,
        use_known_obstacles: bool,
        use_unknown_obstacles: bool,
    ) -> None:
        known_pos = np.random.uniform(size=(self.max_known_obstacles, 2))
        unknown_pos = np.random.uniform(size=(self.max_unknown_obstacles, 2))
        radius = 2 * self.mower_radius + self.obstacle_radius
        assert size_m > 4 * radius
        for n in range(max(self.max_known_obstacles, self.max_unknown_obstacles)):
            if use_known_obstacles and n < self.max_known_obstacles:
                pos_m = 2 * radius + known_pos[n] * (size_m - 4 * radius)
                local_known = self._get_local_neighborhood(
                    known_obstacle_map, pos_m, radius, size_p
                )
                local_unknown = self._get_local_neighborhood(
                    unknown_obstacle_map, pos_m, radius, size_p
                )
                if local_known.sum() == 0 and local_unknown.sum() == 0:
                    cv2.circle(
                        known_obstacle_map,
                        center=np.flip(pos_m * self.pixels_per_meter).astype(np.int32),
                        radius=int(self.obstacle_radius * self.pixels_per_meter),
                        color=1,
                        thickness=cv2.FILLED,
                        lineType=self.line_type,
                    )
            if use_unknown_obstacles and n < self.max_unknown_obstacles:
                pos_m = 2 * radius + unknown_pos[n] * (size_m - 4 * radius)
                local_known = self._get_local_neighborhood(
                    known_obstacle_map, pos_m, radius, size_p
                )
                local_unknown = self._get_local_neighborhood(
                    unknown_obstacle_map, pos_m, radius, size_p
                )
                if local_known.sum() == 0 and local_unknown.sum() == 0:
                    cv2.circle(
                        unknown_obstacle_map,
                        center=np.flip(pos_m * self.pixels_per_meter).astype(np.int32),
                        radius=int(self.obstacle_radius * self.pixels_per_meter),
                        color=1,
                        thickness=cv2.FILLED,
                        lineType=self.line_type,
                    )

    def _get_local_neighborhood_indices(
        self,
        size_p: int,
        pos1_m: np.ndarray,
        pos2_m: np.ndarray,
        radius_m: float,
    ) -> tuple[int, int, int, int]:
        i1 = min(pos1_m[0], pos2_m[0]) - radius_m
        i2 = max(pos1_m[0], pos2_m[0]) + radius_m
        j1 = min(pos1_m[1], pos2_m[1]) - radius_m
        j2 = max(pos1_m[1], pos2_m[1]) + radius_m
        i1 = max(0, min(size_p, int(i1 * self.pixels_per_meter - 10)))
        i2 = max(0, min(size_p, int(i2 * self.pixels_per_meter + 10)))
        j1 = max(0, min(size_p, int(j1 * self.pixels_per_meter - 10)))
        j2 = max(0, min(size_p, int(j2 * self.pixels_per_meter + 10)))
        return i1, i2, j1, j2

    def _get_local_neighborhood(
        self, map_data: np.ndarray, pos_m: np.ndarray, radius_m: float, size_p: int
    ) -> np.ndarray:
        i1, i2, j1, j2 = self._get_local_neighborhood_indices(
            size_p, pos_m, pos_m, radius_m
        )
        return map_data[i1:i2, j1:j2]
