from __future__ import annotations

seed = 7
# Training / optimizer
cnn_dims: int = 256
buffer_size = 500_000
learning_rate = 2e-5
train_freq = 1
gradient_steps = 1
total_timesteps = 1_000_000
total_timesteps = 100

# Feature extractor architecture
grouped_convs: bool = True

# Curriculum / task setup
is_exploration = True
start_level = 1
collision_ends_episode = True

# Robot geometry / task scale
mower_radius = 0.15
lidar_range = mower_radius * 5

# Motion limits
motion_max_lin_vel = 0.26
motion_max_ang_vel = 1.0

# Range sensor
range_num_beams = 24
range_max_distance = lidar_range
range_fov_degrees = 345.0

# Rendering / logging
render_frequency = 1

# Curve-action decoding
action_lidar_scale = 1.0
action_mower_scale = 0.0
action_hybrid_weight = 1.0

info_log_key = "logs"

log_keys = [
    {
        "prefix": "metrics_",
        "title": "metrics",
        "fields": [
            "metrics_truncated",
            "metrics_terminated",
            "metrics_elapsed_steps",
            "metrics_elapsed_time",
            "metrics_non_new_steps",
            "metrics_stuck_steps",
            "metrics_num_collisions",
            "metrics_coverage_in_percent",
            "metrics_overlap_in_percent",
            "metrics_global_total_variation",
        ],
    },
    {
        "prefix": "metrics_",
        "title": "rewards",
        "fields": [
            "metrics_reward_area",
            "metrics_reward_overlap",
            "metrics_reward_tv",
            "metrics_reward_frontier",
            "metrics_reward_turn",
            "metrics_reward_collision",
            "metrics_reward_goal",
            "metrics_reward_truncation",
            "metrics_reward_constant",
            "metrics_reward_total",
        ],
    },
    {
        "prefix": "episode_",
        "title": "per_episode",
        "fields": [
            "episode_level",
            "episode_filename",
        ],
    },
]

info_log_file = "info_logs.csv"

runtime_config_path = "config.json"

logger_formats = ["log"]

experiments_dir_name = "experiments"

env_type = ""
# env_type = "curve"
# env_type="mower"

resume_from_checkpoint_file = ""
reset_num_timesteps = False


def start_training(runtime_manager=None):

    import os
    from datetime import datetime
    from pathlib import Path

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common import logger

    from callbacks.info_logging import InfoLoggingCallback
    from callbacks.checkpoint import FnCheckpointCallback
    from callbacks.stopper import StopperCallback

    from envs.curve_env import CurveActionEnv
    from envs.mower_env import MowerEnv

    from grid.grid_collision import GridCollision
    from grid.grid_curriculum import GridCurriculumManager
    from grid.grid_lidar import GridLidarRanging
    from grid.grid_motion import GridMotion

    from runtime.runtime_manager import RuntimeManager

    from sgcnn.architectures import StackedMapFeaturesExtractor
    from sgcnn.coverage_mapper import RangeCoverageMapper

    root_dir = Path.cwd()

    experiment_dir = os.path.join(
        root_dir,
        experiments_dir_name,
        "curve",
        datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
    )

    render_dir = os.path.join(experiment_dir, "renders")

    checkpoints_dir = os.path.join(experiment_dir, "checks")

    info_log_path = os.path.join(experiment_dir, info_log_file)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    runtime = (
        runtime_manager
        if isinstance(runtime_manager, RuntimeManager)
        else RuntimeManager()
    )

    runtime.set_directories(
        experiment_dir=experiment_dir,
        renders_dir=render_dir,
        info_file=info_log_path,
    )

    curriculum = GridCurriculumManager(
        start_level=start_level,
        exploration=is_exploration,
        mower_radius=mower_radius,
    )

    motion = GridMotion(
        motion_max_lin_vel=motion_max_lin_vel,
        motion_max_ang_vel=motion_max_ang_vel,
    )

    ranging = GridLidarRanging(
        range_num_beams=range_num_beams,
        range_max_distance=lidar_range,
        range_fov_degrees=range_fov_degrees,
    )

    collision = GridCollision()

    mapper = RangeCoverageMapper(
        mapper_exploration=is_exploration,
    )

    cenv = None
    if env_type == "curve":
        cenv = CurveActionEnv(
            curriculum=curriculum,
            motion=motion,
            ranging=ranging,
            collision=collision,
            mapper=mapper,
            episode_collision_ends=collision_ends_episode,
            render_dir=render_dir,
            render_frequency=render_frequency,
            action_lidar_scale=action_lidar_scale,
            action_mower_scale=action_mower_scale,
            action_hybrid_weight=action_hybrid_weight,
            config_can_render=lambda: runtime.can_render,
            info_log_key=info_log_key,
        )
    elif env_type == "mower":
        cenv = MowerEnv(
            curriculum=curriculum,
            motion=motion,
            ranging=ranging,
            collision=collision,
            mapper=mapper,
            episode_collision_ends=collision_ends_episode,
            render_dir=render_dir,
            render_frequency=render_frequency,
            config_can_render=lambda: runtime.can_render,
            info_log_key=info_log_key,
        )

    if cenv is None:
        raise Exception("env_type must be either 'curve' or 'mower'")

    info_logger = InfoLoggingCallback(
        filepath=info_log_path,
        get_buffer_size=lambda: runtime.csv_buffer_size,
        log_keys=log_keys,
        info_key=info_log_key,
    )

    stopper = StopperCallback(
        stop_training_fn=lambda: runtime.end_training,
    )

    custom_logger = logger.configure(
        folder=experiment_dir,
        format_strings=logger_formats,
    )

    callbacks = CallbackList(
        [
            FnCheckpointCallback(
                save_freq_fn=lambda: runtime.checkpoint_save_freq,
                save_path=checkpoints_dir,
                name_prefix=env_type,
                save_replay_buffer_fn=lambda: runtime.checkpoint_save_replay_buffer,
                save_vecnormalize_fn=lambda: runtime.checkpoint_save_vecnormalize,
                use_static_files=lambda: runtime.checkpoint_use_static_files,
            ),
            info_logger,
            stopper,
        ]
    )

    vec_env = DummyVecEnv([lambda: cenv])

    policy_kwargs = dict(
        net_arch=dict(pi=[cnn_dims, cnn_dims], qf=[cnn_dims, cnn_dims]),
        features_extractor_class=StackedMapFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=cnn_dims,
            grouped_convs=grouped_convs,
            lidar_rays=ranging.range_num_beams,
            map_size=mapper.input_size,
            num_maps=mapper.num_maps,
            stacks=cenv.observation_stacks,
            frontier_observation=mapper.frontier_observation,
        ),
    )

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        seed=seed,
    )

    if resume_from_checkpoint_file:
        model = SAC.load(
            resume_from_checkpoint_file,
            env=vec_env,
        )

    model.set_logger(custom_logger)

    runtime.start()

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )
