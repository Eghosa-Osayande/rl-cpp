from __future__ import annotations

import math
import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sgcnn.interfaces import (
    CollisionInterface,
    CurriculumManager,
    EpisodeSpec,
    MotionInterface,
    MotionStepResult,
    PoseState,
    RangeState,
    RangingInterface,
)

from sgcnn.coverage_mapper import RangeCoverageMapper
from sgcnn.utils import (
    angle_diff,
    collision_reward,
    make_episode_result,
    make_history_buffer,
    normalize_ranges,
    read_history,
    write_history,
)


class MowerEnv(gym.Env):
    def __init__(
        self,
        *,
        curriculum: CurriculumManager,
        motion: MotionInterface,
        ranging: RangingInterface | None,
        collision: CollisionInterface,
        mapper: RangeCoverageMapper,
        observation_stacks: int = 1,
        observation_action_history: int = 0,
        episode_step_size: float = 0.5,
        episode_max_steps: int | None = None,
        episode_max_non_new_steps: int | None = 1000,
        episode_collision_ends: bool = False,
        episode_flip_when_stuck: bool = False,
        episode_max_stuck_steps: int = 5,
        reward_goal_coverage: float = 0.0,
        reward_collision_wall: float = -10.0,
        reward_collision_obstacle: float = -10.0,
        reward_newly_visited_scale: float = 1.0,
        reward_newly_visited_max: float = 2.0,
        reward_overlap_scale: float = 0.0,
        reward_overlap_max: float = 5.0,
        reward_overlap_always: bool = False,
        reward_local_tv_scale: float = 1.0,
        reward_local_tv_max: float = 5.0,
        reward_global_tv_scale: float = 0.0,
        reward_global_tv_max: float = 5.0,
        reward_use_known_obstacles_in_tv: bool = True,
        reward_use_unknown_obstacles_in_tv: bool = True,
        reward_frontier_scale: float = 0.0,
        reward_frontier_max: float = 5.0,
        reward_turn_scale: float = 0.0,
        reward_constant: float = -0.1,
        reward_constant_always: bool = True,
        reward_truncation_scale: float = 0.0,
        render_verbose: bool = False,
        render_dir: str | None = None,
        render_frequency: int = 0,
        action_constant_lin_vel: bool = True,
        action_steering_limits_lin_vel: bool = True,
        action_reverse_speed_scale: float = 0.5,
        config_can_render=lambda: True,
        info_log_key="logs",
    ):
        super().__init__()
        self.info_log_key = info_log_key
        self.config_can_render = config_can_render
        self.curriculum = curriculum
        self.motion = motion
        self.ranging = ranging
        self.collision = collision
        self.mapper = mapper
        self.observation_stacks = int(observation_stacks)
        self.observation_action_history = int(observation_action_history)
        self.episode_step_size = float(episode_step_size)
        self.episode_max_steps = episode_max_steps
        self.episode_max_non_new_steps = episode_max_non_new_steps
        self.episode_collision_ends = bool(episode_collision_ends)
        self.episode_flip_when_stuck = bool(episode_flip_when_stuck)
        self.episode_max_stuck_steps = int(episode_max_stuck_steps)
        self.reward_goal_coverage = float(reward_goal_coverage)
        self.reward_collision_wall = float(reward_collision_wall)
        self.reward_collision_obstacle = float(reward_collision_obstacle)
        self.reward_newly_visited_scale = float(reward_newly_visited_scale)
        self.reward_newly_visited_max = float(reward_newly_visited_max)
        self.reward_overlap_scale = float(reward_overlap_scale)
        self.reward_overlap_max = float(reward_overlap_max)
        self.reward_overlap_always = bool(reward_overlap_always)
        self.reward_local_tv_scale = float(reward_local_tv_scale)
        self.reward_local_tv_max = float(reward_local_tv_max)
        self.reward_global_tv_scale = float(reward_global_tv_scale)
        self.reward_global_tv_max = float(reward_global_tv_max)
        self.reward_use_known_obstacles_in_tv = bool(reward_use_known_obstacles_in_tv)
        self.reward_use_unknown_obstacles_in_tv = bool(
            reward_use_unknown_obstacles_in_tv
        )
        self.reward_frontier_scale = float(reward_frontier_scale)
        self.reward_frontier_max = float(reward_frontier_max)
        self.reward_turn_scale = float(reward_turn_scale)
        self.reward_constant = float(reward_constant)
        self.reward_constant_always = bool(reward_constant_always)
        self.reward_truncation_scale = float(reward_truncation_scale)
        self.render_verbose = bool(render_verbose)
        self.render_dir = render_dir
        self.render_frequency = int(render_frequency)
        self.include_artifacts_in_info = True
        self.action_constant_lin_vel = bool(action_constant_lin_vel)
        self.action_steering_limits_lin_vel = bool(action_steering_limits_lin_vel)
        self.action_reverse_speed_scale = float(action_reverse_speed_scale)

        action_dim = 1 if self.action_constant_lin_vel else 2
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        map_obs_shape = (
            self.observation_stacks,
            self.mapper.num_maps,
            self.mapper.input_size,
            self.mapper.input_size,
        )
        observation_space_dict: dict[str, spaces.Space] = {
            "coverage": spaces.Box(
                low=0, high=1, shape=map_obs_shape, dtype=np.float32
            ),
            "obstacles": spaces.Box(
                low=0, high=1, shape=map_obs_shape, dtype=np.float32
            ),
        }
        self._include_lidar = self.ranging is not None
        if self._include_lidar:
            observation_space_dict["lidar"] = spaces.Box(
                low=0,
                high=1,
                shape=(self.observation_stacks, self.ranging.range_num_beams),
                dtype=np.float32,
            )
        if self.mapper.frontier_observation:
            observation_space_dict["frontier"] = spaces.Box(
                low=0,
                high=1,
                shape=map_obs_shape,
                dtype=np.float32,
            )
        if self.observation_action_history > 0:
            observation_space_dict["action"] = spaces.Box(
                low=-1,
                high=1,
                shape=(self.observation_action_history, action_dim),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_space_dict)
        self.current_episode = 0
        self.level = self.curriculum.get_progress().level

    def reset(self, *, seed=None, options=None):
        if hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)
            except TypeError:
                super().reset()
        del options

        self.current_episode += 1
        self.elapsed_steps = 0
        self.elapsed_time = 0.0
        self.non_new_steps = 0
        self.stuck_steps = 0
        self.num_collisions = 0
        self._episode = self.curriculum.next_episode()
        self.level = self.curriculum.get_progress().level
        self.collision.reset(self._episode)
        pose_state = self._spawn_valid_pose_state(self._episode)
        range_state = (
            self.ranging.reset(self._episode, pose_state)
            if self.ranging is not None
            else None
        )
        mapper_state = self.mapper.reset(self._episode, pose_state, range_state)
        self._pose_state = pose_state
        self._range_state = range_state
        self._mapper_state = mapper_state
        self._coverage_history = make_history_buffer(
            mapper_state.coverage_state_projected, self.observation_stacks
        )
        self._obstacle_history = make_history_buffer(
            mapper_state.obstacle_state_projected, self.observation_stacks
        )
        if (
            self.mapper.frontier_observation
            and mapper_state.frontier_state_projected is not None
        ):
            self._frontier_history = make_history_buffer(
                mapper_state.frontier_state_projected,
                self.observation_stacks,
            )
        else:
            self._frontier_history = None
        if self._include_lidar and range_state is not None:
            lidar_obs = normalize_ranges(
                self._range_distances(range_state), range_state.max_distance
            )
            self._lidar_history = make_history_buffer(
                lidar_obs, self.observation_stacks
            )
        else:
            self._lidar_history = None
        if self.observation_action_history > 0:
            self._action_history = np.zeros(
                self.observation_space["action"].shape, dtype=np.float32
            )
        else:
            self._action_history = None
        observation = self._stacked_observation()
        info = self._build_info(
            reward_area=0.0,
            reward_overlap=0.0,
            reward_tv=0.0,
            reward_frontier=0.0,
            reward_turn=0.0,
            reward_collision=0.0,
            reward_goal=0.0,
            reward_truncation=0.0,
            reward_constant=0.0,
            reward_total=0.0,
            non_new=False,
            terminated=False,
            truncated=False,
        )
        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        previous_pose_state = self._pose_state
        motion_result = self._step_motion(action)
        collision_result = self.collision.resolve_path(
            [previous_pose_state, motion_result.pose_state]
        )

        terminated = False
        reward_collision = 0.0
        if collision_result.collided:
            reward_collision = collision_reward(
                collision_result.collision_type,
                self.reward_collision_wall,
                self.reward_collision_obstacle,
            )
            self.num_collisions += 1
            self.stuck_steps += 1
            self._pose_state = self.motion.set_pose(
                collision_result.pose_state.position_m,
                collision_result.pose_state.heading,
            )
            if self.episode_collision_ends:
                terminated = True
        else:
            self.stuck_steps = 0
            self._pose_state = motion_result.pose_state

        if (
            self.episode_flip_when_stuck
            and self.stuck_steps >= self.episode_max_stuck_steps
        ):
            self.stuck_steps = 0
            self._pose_state = self.motion.set_pose(
                self._pose_state.position_m,
                (self._pose_state.heading + math.pi) % (2 * math.pi),
            )

        self._range_state = (
            self.ranging.scan(self._pose_state, dt=self.episode_step_size)
            if self.ranging is not None
            else None
        )
        map_step_state = self.mapper.step(
            previous_pose_state, self._pose_state, self._range_state
        )
        self._mapper_state = map_step_state.mapper_state

        if map_step_state.newly_visited_sum > 0:
            self.non_new_steps = 0

        write_history(
            self._coverage_history,
            self.elapsed_steps,
            self._mapper_state.coverage_state_projected,
        )
        write_history(
            self._obstacle_history,
            self.elapsed_steps,
            self._mapper_state.obstacle_state_projected,
        )
        if (
            self._frontier_history is not None
            and self._mapper_state.frontier_state_projected is not None
        ):
            write_history(
                self._frontier_history,
                self.elapsed_steps,
                self._mapper_state.frontier_state_projected,
            )
        if self._lidar_history is not None and self._range_state is not None:
            lidar_obs = normalize_ranges(
                self._range_distances(self._range_state), self._range_state.max_distance
            )
            write_history(self._lidar_history, self.elapsed_steps, lidar_obs)
        if self._action_history is not None:
            write_history(
                self._action_history, self.elapsed_steps, action.astype(np.float32)
            )

        reward_area = 0.0
        reward_overlap = 0.0
        if not collision_result.collided:
            width_m = 2 * self._episode.mower_radius
            if self.mapper.exploration and self._range_state is not None:
                width_m = 2 * self._range_state.max_distance
            length_m = self.motion.max_lin_vel * self.episode_step_size
            max_newly_visited_sum = max(
                1e-9, width_m * length_m * self.mapper.pixels_per_meter**2
            )
            reward_area = (
                self.reward_newly_visited_scale
                * map_step_state.newly_visited_dilated_sum
            )
            reward_area /= max_newly_visited_sum
            reward_area = min(reward_area, self.reward_newly_visited_max)
            if self.reward_overlap_always or reward_area == 0:
                reward_overlap = float(
                    (
                        map_step_state.local_coverage_diff
                        * map_step_state.local_overlap_before
                    ).sum()
                )
                reward_overlap *= self.reward_overlap_scale / max_newly_visited_sum
                reward_overlap = -min(reward_overlap, self.reward_overlap_max)

        reward_tv = 0.0
        if (
            not collision_result.collided
            and self.motion.max_lin_vel > 0
            and self.episode_step_size > 0
        ):
            reward_itv = -map_step_state.total_variation_diff
            reward_itv *= (
                self.mapper.meters_per_pixel
                / self.episode_step_size
                / self.motion.max_lin_vel
            )
            reward_itv /= 2.5
            reward_itv *= self.reward_local_tv_scale
            reward_itv = np.sign(reward_itv) * min(
                abs(reward_itv), self.reward_local_tv_max
            )
            reward_gtv = 0.0
            if self._mapper_state.coverage_in_pixels > 0:
                reward_gtv = -self._mapper_state.global_total_variation
                reward_gtv /= math.sqrt(self._mapper_state.coverage_in_pixels)
                reward_gtv /= 4
                reward_gtv *= self.reward_global_tv_scale
                reward_gtv = np.sign(reward_gtv) * min(
                    abs(reward_gtv), self.reward_global_tv_max
                )
            reward_tv = float(reward_itv + reward_gtv)

        reward_frontier = 0.0
        if (
            self.mapper.frontier_observation
            and not collision_result.collided
            and map_step_state.newly_visited_sum == 0
            and self.ranging is not None
            and self.motion.max_lin_vel > 0
            and self.episode_step_size > 0
        ):
            reward_frontier = (
                map_step_state.frontier_distance_before
                - map_step_state.frontier_distance_after
            )
            reward_frontier *= self.reward_frontier_scale
            reward_frontier /= self.motion.max_lin_vel * self.episode_step_size
            reward_frontier = max(reward_frontier, -self.reward_frontier_max)
            reward_frontier = min(0.0, reward_frontier)

        reward_turn = 0.0
        if (
            not collision_result.collided
            and self.motion.max_ang_vel > 0
            and self.episode_step_size > 0
        ):
            reward_turn = -self.reward_turn_scale * min(
                angle_diff(self._pose_state.heading, previous_pose_state.heading)
                / (self.motion.max_ang_vel * self.episode_step_size),
                1.0,
            )

        reward_goal = 0.0
        reached_goal = False
        curriculum_elapsed_steps = self.elapsed_steps
        if (
            not terminated
            and self._mapper_state.coverage_in_percent >= self._episode.goal_coverage
        ):
            reached_goal = True
            reward_goal = self.reward_goal_coverage
            terminated = True

        self.elapsed_steps += 1
        self.non_new_steps += 1
        self.elapsed_time = self.elapsed_steps * self.episode_step_size

        truncated = False
        reward_truncation = 0.0
        non_new = False
        if (
            self.episode_max_steps is not None
            and not terminated
            and self.elapsed_steps >= self.episode_max_steps
        ):
            reward_truncation = -self.reward_truncation_scale * (
                1 - self._mapper_state.coverage_in_percent
            )
            truncated = True
        if (
            self.episode_max_non_new_steps is not None
            and not terminated
            and not truncated
            and self.non_new_steps >= self.episode_max_non_new_steps
        ):
            reward_truncation = -self.reward_truncation_scale * (
                1 - self._mapper_state.coverage_in_percent
            )
            truncated = True
            non_new = True

        reward_constant = (
            self.reward_constant
            if (self.reward_constant_always or map_step_state.newly_visited_sum == 0)
            else 0.0
        )
        reward_total = (
            reward_area
            + reward_overlap
            + reward_tv
            + reward_frontier
            + reward_turn
            + reward_collision
            + reward_goal
            + reward_truncation
            + reward_constant
        )

        if terminated or truncated:
            self.curriculum.on_episode_end(
                make_episode_result(
                    episode=self._episode,
                    reached_goal=reached_goal,
                    elapsed_steps=curriculum_elapsed_steps,
                    coverage_in_percent=self._mapper_state.coverage_in_percent,
                    truncated=truncated,
                    metrics={
                        "num_collisions": self.num_collisions,
                        "coverage_in_pixels": self._mapper_state.coverage_in_pixels,
                    },
                )
            )
            self.level = self.curriculum.get_progress().level

        observation = self._stacked_observation()
        info = self._build_info(
            reward_area=reward_area,
            reward_overlap=reward_overlap,
            reward_tv=reward_tv,
            reward_frontier=reward_frontier,
            reward_turn=reward_turn,
            reward_collision=reward_collision,
            reward_goal=reward_goal,
            reward_truncation=reward_truncation,
            reward_constant=reward_constant,
            reward_total=reward_total,
            non_new=non_new,
            terminated=terminated,
            truncated=truncated,
        )
        if self.render_verbose:
            print(
                "Step:",
                f"{self.elapsed_steps},",
                f"Coverage:{round(100 * self._mapper_state.coverage_in_percent, 2):>6},",
                f"Reward:{round(reward_total, 2):>6}",
            )
        if (
            self.render_frequency > 0
            and self.elapsed_steps % self.render_frequency == 0
        ):
            self.render()
        return observation, reward_total, terminated, truncated, info

    def render(self):
        if self.render_dir is not None and self.config_can_render():
            image = self.mapper.render()
            os.makedirs(self.render_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(self.render_dir, f"{self.current_episode:07d}.png"), image
            )
            return image

        return None

    def close(self):
        return None

    def _spawn_valid_pose_state(self, episode: EpisodeSpec) -> PoseState:
        if episode.spawn_position_m is not None:
            return self.motion.reset(episode)
        for _ in range(1000):
            pose_state = self.motion.reset(episode)
            if not self.collision.resolve_path([pose_state]).collided:
                return pose_state
        raise RuntimeError("Could not sample a collision-free start pose.")

    def _range_distances(self, range_state: RangeState) -> np.ndarray:
        return np.asarray(range_state.samples[:, 0], dtype=np.float32)

    def _step_motion(self, action: np.ndarray) -> MotionStepResult:
        if self.action_constant_lin_vel:
            throttle = 1.0
            steering = float(action[0])
        else:
            throttle = float(action[0])
            steering = float(action[1])
        throttle = float(np.clip(throttle, -1, 1))
        steering = float(np.clip(steering, -1, 1))
        linear_velocity = throttle
        if self.action_steering_limits_lin_vel:
            linear_velocity *= 1 - abs(steering)
        linear_velocity *= self.motion.max_lin_vel
        if linear_velocity < 0:
            linear_velocity *= self.action_reverse_speed_scale
        angular_velocity = steering * self.motion.max_ang_vel
        return self.motion.step_linear_angular(
            linear_velocity, angular_velocity, self.episode_step_size
        )

    def _stacked_observation(self) -> dict[str, np.ndarray]:
        observation = {
            "coverage": read_history(self._coverage_history, self.elapsed_steps),
            "obstacles": read_history(self._obstacle_history, self.elapsed_steps),
        }
        if self._frontier_history is not None:
            observation["frontier"] = read_history(
                self._frontier_history, self.elapsed_steps
            )
        if self._lidar_history is not None:
            observation["lidar"] = read_history(self._lidar_history, self.elapsed_steps)
        if self._action_history is not None:
            observation["action"] = read_history(
                self._action_history, self.elapsed_steps
            )
        return observation

    def _build_info(
        self,
        *,
        reward_area: float,
        reward_overlap: float,
        reward_tv: float,
        reward_frontier: float,
        reward_turn: float,
        reward_collision: float,
        reward_goal: float,
        reward_truncation: float,
        reward_constant: float,
        reward_total: float,
        non_new: bool,
        terminated: bool,
        truncated: bool,
    ) -> dict:
        logs = {
            "metrics_dt_step": 0,
            "metrics_dt_lidar": 0,
            "metrics_dt_other": 0,
            "metrics_elapsed_steps": self.elapsed_steps,
            "metrics_elapsed_time": self.elapsed_time,
            "metrics_non_new_steps": self.non_new_steps,
            "metrics_stuck_steps": self.stuck_steps,
            "metrics_num_collisions": self.num_collisions,
            "metrics_coverage_in_pixels": self._mapper_state.coverage_in_pixels,
            "metrics_coverage_in_m2": self._mapper_state.coverage_in_m2,
            "metrics_coverage_in_percent": self._mapper_state.coverage_in_percent,
            "metrics_overlap_in_pixels": self._mapper_state.overlap_in_pixels,
            "metrics_overlap_in_m2": self._mapper_state.overlap_in_m2,
            "metrics_overlap_in_percent": self._mapper_state.overlap_in_percent,
            "metrics_global_total_variation": self._mapper_state.global_total_variation,
            "metrics_frontier_distance": self._mapper_state.frontier_distance,
            "metrics_position_x": float(self._pose_state.position_m[0]),
            "metrics_position_y": float(self._pose_state.position_m[1]),
            "metrics_heading": self._pose_state.heading,
            "metrics_reward_area": reward_area,
            "metrics_reward_overlap": reward_overlap,
            "metrics_reward_tv": reward_tv,
            "metrics_reward_frontier": reward_frontier,
            "metrics_reward_turn": reward_turn,
            "metrics_reward_collision": reward_collision,
            "metrics_reward_goal": reward_goal,
            "metrics_reward_truncation": reward_truncation,
            "metrics_reward_constant": reward_constant,
            "metrics_reward_total": reward_total,
            "metrics_non_new": non_new,
            "metrics_truncated": truncated,
            "metrics_terminated": terminated,
            "episode_episode_index": self.current_episode,
            "episode_level": self.level,
            "episode_eval_mode": self._episode.eval_mode,
            "episode_filename": self._episode.filename,
            "episode_current_map": self._episode.current_map,
            "episode_use_floor_plans": self._episode.use_floor_plans,
            "episode_use_known_obstacles": self._episode.use_known_obstacles,
            "episode_use_unknown_obstacles": self._episode.use_unknown_obstacles,
        }

        return {self.info_log_key: logs}
