#!/usr/bin/env python3

"""
003-01 — PPO Baseline
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import (
    check_env,
)
from stable_baselines3.common.monitor import (
    Monitor,
)

from sesame_rl_env import SesameRLEnv


# ============================================================
# CONFIG
# ============================================================

SEED = 1000

TOTAL_TIMESTEPS = 300_000

RESULT_DIR = Path(
    "003_rl_results/01_ppo_baseline"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK ENV
# ============================================================

print(
    "=== 003-01 PPO BASELINE ==="
)

test_env = SesameRLEnv()

check_env(
    test_env,
    warn=True,
)

test_env.close()

print(
    "Environment check : PASS"
)


# ============================================================
# TRAINING ENV
# ============================================================

env = Monitor(
    SesameRLEnv(),
    filename=str(
        RESULT_DIR
        / "monitor"
    ),
)

env.reset(
    seed=SEED
)


# ============================================================
# PPO
# ============================================================

model = PPO(

    "MlpPolicy",

    env,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=64,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    seed=SEED,

    verbose=1,

    device="auto",
)


# ============================================================
# TRAIN
# ============================================================

print()
print(
    f"Training PPO for "
    f"{TOTAL_TIMESTEPS} steps..."
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS
)


# ============================================================
# SAVE
# ============================================================

MODEL_PATH = (
    RESULT_DIR
    / "ppo_baseline"
)

model.save(
    str(
        MODEL_PATH
    )
)

env.close()

print()
print(
    "003-01 COMPLETE"
)

print(
    f"Model saved:"
)

print(
    f"{MODEL_PATH}.zip"
)

"""
003-01 — PPO Baseline Training
==============================

First learned locomotion experiment for Sesame-RL.

Purpose
-------
Determine whether PPO can learn useful forward locomotion
from the validated 003-00 environment.

Important:
- Environment is NOT redefined here.
- Reward V1 remains frozen.
- No domain randomization.
- No curriculum.
- No reward sweep.
- No sim-to-real noise.
- No firmware imitation.

Baseline comparison later:

    Random Policy
        vs
    Firmware Policy
        vs
    PPO Policy
"""
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import sys

import numpy as np

import torch

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import (
    Monitor,
)

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from stable_baselines3.common.env_checker import (
    check_env,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)


from sesame_rl_env import (
    SesameRLEnv,
    CONTROL_DT,
    MAX_EPISODE_STEPS,
    ACTION_SCALE,
    W_FORWARD,
    W_LATERAL,
    W_YAW,
    W_UPRIGHT,
    W_ACTION_RATE,
    ALIVE_REWARD,
    FALL_PENALTY,
)


# ============================================================
# TRAINING CONFIG
# ============================================================

DEFAULT_TOTAL_TIMESTEPS = 300_000

LEARNING_RATE = 3e-4

N_STEPS = 2048

BATCH_SIZE = 64

N_EPOCHS = 10

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.20

ENT_COEF = 0.01

VF_COEF = 0.50

MAX_GRAD_NORM = 0.50


# ============================================================
# EVALUATION
# ============================================================

DEFAULT_EVAL_FREQ = 10_000

EVAL_EPISODES = 5

DEFAULT_CHECKPOINT_FREQ = 50_000


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR.parent
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "003_rl_results"
    / "01_ppo_baseline"
)

MODEL_DIR = (
    RESULT_ROOT
    / "models"
)

CHECKPOINT_DIR = (
    RESULT_ROOT
    / "checkpoints"
)

BEST_MODEL_DIR = (
    RESULT_ROOT
    / "best_model"
)

EVAL_DIR = (
    RESULT_ROOT
    / "eval"
)

MONITOR_DIR = (
    RESULT_ROOT
    / "monitor"
)

TENSORBOARD_DIR = (
    RESULT_ROOT
    / "tensorboard"
)

for directory in [
    RESULT_ROOT,
    MODEL_DIR,
    CHECKPOINT_DIR,
    BEST_MODEL_DIR,
    EVAL_DIR,
    MONITOR_DIR,
    TENSORBOARD_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# ENV FACTORY
# ============================================================

def make_training_env(
    seed,
):

    def _factory():

        env = SesameRLEnv(
            render_mode=None
        )

        env = Monitor(
            env,
            filename=str(
                MONITOR_DIR
                / "train"
            ),
        )

        env.reset(
            seed=seed
        )

        return env

    return _factory


def make_eval_env(
    seed,
):

    def _factory():

        env = SesameRLEnv(
            render_mode=None
        )

        env = Monitor(
            env,
            filename=str(
                MONITOR_DIR
                / "eval"
            ),
        )

        env.reset(
            seed=seed
        )

        return env

    return _factory


# ============================================================
# CONFIG EXPORT
# ============================================================

def save_experiment_config(
    args,
):

    try:

        import stable_baselines3

        sb3_version = (
            stable_baselines3.__version__
        )

    except Exception:

        sb3_version = (
            "unknown"
        )

    try:

        import gymnasium

        gymnasium_version = (
            gymnasium.__version__
        )

    except Exception:

        gymnasium_version = (
            "unknown"
        )

    try:

        import mujoco

        mujoco_version = (
            mujoco.__version__
        )

    except Exception:

        mujoco_version = (
            "unknown"
        )

    config = {

        # ----------------------------------------------------
        # experiment
        # ----------------------------------------------------

        "experiment":
            "003-01 PPO Baseline",

        "seed":
            args.seed,

        "total_timesteps":
            args.timesteps,

        "device":
            args.device,

        # ----------------------------------------------------
        # PPO
        # ----------------------------------------------------

        "algorithm":
            "PPO",

        "policy":
            "MlpPolicy",

        "learning_rate":
            LEARNING_RATE,

        "n_steps":
            N_STEPS,

        "batch_size":
            BATCH_SIZE,

        "n_epochs":
            N_EPOCHS,

        "gamma":
            GAMMA,

        "gae_lambda":
            GAE_LAMBDA,

        "clip_range":
            CLIP_RANGE,

        "ent_coef":
            ENT_COEF,

        "vf_coef":
            VF_COEF,

        "max_grad_norm":
            MAX_GRAD_NORM,

        # ----------------------------------------------------
        # environment
        # ----------------------------------------------------

        "control_dt":
            CONTROL_DT,

        "episode_steps":
            MAX_EPISODE_STEPS,

        "action_scale_rad":
            ACTION_SCALE,

        # ----------------------------------------------------
        # reward
        # ----------------------------------------------------

        "reward": {

            "W_FORWARD":
                W_FORWARD,

            "W_LATERAL":
                W_LATERAL,

            "W_YAW":
                W_YAW,

            "W_UPRIGHT":
                W_UPRIGHT,

            "W_ACTION_RATE":
                W_ACTION_RATE,

            "ALIVE_REWARD":
                ALIVE_REWARD,

            "FALL_PENALTY":
                FALL_PENALTY,
        },

        # ----------------------------------------------------
        # evaluation
        # ----------------------------------------------------

        "eval_freq":
            args.eval_freq,

        "eval_episodes":
            EVAL_EPISODES,

        "checkpoint_freq":
            args.checkpoint_freq,

        # ----------------------------------------------------
        # software
        # ----------------------------------------------------

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "torch":
            torch.__version__,

        "stable_baselines3":
            sb3_version,

        "gymnasium":
            gymnasium_version,

        "mujoco":
            mujoco_version,
    }

    path = (
        RESULT_ROOT
        / "config.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    print(
        f"Config saved      : {path}"
    )


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def validate_environment():

    print()
    print(
        "=== 003-01 ENVIRONMENT VALIDATION ==="
    )

    env = SesameRLEnv(
        render_mode=None
    )

    check_env(
        env,
        warn=True,
    )

    obs, info = env.reset(
        seed=1000
    )

    print(
        f"Observation shape : "
        f"{obs.shape}"
    )

    print(
        f"Action shape      : "
        f"{env.action_space.shape}"
    )

    print(
        f"Initial height    : "
        f"{info['body_height']:.5f} m"
    )

    print(
        f"Initial upright   : "
        f"{info['upright']:.5f}"
    )

    print(
        "SB3 check_env     : PASS"
    )

    env.close()


# ============================================================
# FINAL DETERMINISTIC EVALUATION
# ============================================================

def evaluate_policy_detailed(
    model,
    episodes=5,
    seed=2000,
    render=False,
):

    print()
    print(
        "=== 003-01 FINAL POLICY EVALUATION ==="
    )

    env = SesameRLEnv(
        render_mode=(
            "human"
            if render
            else None
        )
    )

    step_rows = []

    episode_rows = []

    for episode in range(
        episodes
    ):

        obs, info = env.reset(
            seed=(
                seed
                + episode
            )
        )

        total_reward = 0.0

        forward_values = []

        forward_abs_values = []

        lateral_values = []

        yaw_values = []

        upright_values = []

        height_values = []

        action_delta_values = []

        previous_action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

        terminated = False

        truncated = False

        last_step = 0

        # ----------------------------------------------------
        # start position
        # ----------------------------------------------------

        start_position = (
            env.data.xpos[
                env.base_body_id
            ].copy()
        )

        path_length = 0.0

        previous_position = (
            start_position.copy()
        )

        for step in range(
            MAX_EPISODE_STEPS
        ):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += (
                reward
            )

            forward = float(
                info[
                    "forward_velocity"
                ]
            )

            lateral = float(
                info[
                    "lateral_velocity"
                ]
            )

            yaw_rate = float(
                info[
                    "yaw_rate"
                ]
            )

            upright = float(
                info[
                    "upright"
                ]
            )

            height = float(
                info[
                    "body_height"
                ]
            )

            action_delta = float(
                np.mean(
                    (
                        action
                        - previous_action
                    ) ** 2
                )
            )

            current_position = (
                env.data.xpos[
                    env.base_body_id
                ].copy()
            )

            planar_step = float(
                np.linalg.norm(
                    current_position[
                        :2
                    ]
                    - previous_position[
                        :2
                    ]
                )
            )

            path_length += (
                planar_step
            )

            previous_position = (
                current_position.copy()
            )

            forward_values.append(
                forward
            )

            forward_abs_values.append(
                abs(
                    forward
                )
            )

            lateral_values.append(
                abs(
                    lateral
                )
            )

            yaw_values.append(
                abs(
                    yaw_rate
                )
            )

            upright_values.append(
                upright
            )

            height_values.append(
                height
            )

            action_delta_values.append(
                action_delta
            )

            step_row = {

                "episode":
                    episode,

                "step":
                    step,

                "time":
                    step
                    * CONTROL_DT,

                "reward":
                    float(
                        reward
                    ),

                "forward_velocity":
                    forward,

                "lateral_velocity":
                    lateral,

                "yaw_rate_rad_s":
                    yaw_rate,

                "yaw_rate_deg_s":
                    float(
                        np.rad2deg(
                            yaw_rate
                        )
                    ),

                "upright":
                    upright,

                "body_height":
                    height,

                "action_delta_mse":
                    action_delta,

                "terminated":
                    int(
                        terminated
                    ),

                "truncated":
                    int(
                        truncated
                    ),
            }

            for i in range(
                len(action)
            ):

                step_row[
                    f"action_{i}"
                ] = float(
                    action[i]
                )

            step_rows.append(
                step_row
            )

            previous_action = (
                np.asarray(
                    action,
                    dtype=np.float32,
                ).copy()
            )

            last_step = step

            if (
                terminated
                or truncated
            ):

                break

        # ----------------------------------------------------
        # final position
        # ----------------------------------------------------

        end_position = (
            env.data.xpos[
                env.base_body_id
            ].copy()
        )

        displacement = float(
            np.linalg.norm(
                end_position[
                    :2
                ]
                - start_position[
                    :2
                ]
            )
        )

        if path_length > 1e-9:

            path_efficiency = (
                displacement
                / path_length
            )

        else:

            path_efficiency = 0.0

        mean_forward = float(
            np.mean(
                forward_values
            )
        )

        mean_abs_forward = float(
            np.mean(
                forward_abs_values
            )
        )

        mean_abs_lateral = float(
            np.mean(
                lateral_values
            )
        )

        mean_abs_yaw = float(
            np.mean(
                yaw_values
            )
        )

        directional_ratio = (
            mean_abs_forward
            /
            (
                mean_abs_forward
                + mean_abs_lateral
                + 1e-9
            )
        )

        episode_row = {

            "episode":
                episode,

            "steps":
                last_step + 1,

            "duration":
                (
                    last_step + 1
                )
                * CONTROL_DT,

            "total_reward":
                total_reward,

            "mean_forward_velocity":
                mean_forward,

            "mean_abs_forward_velocity":
                mean_abs_forward,

            "mean_abs_lateral_velocity":
                mean_abs_lateral,

            "directional_ratio":
                directional_ratio,

            "mean_abs_yaw_rate_rad_s":
                mean_abs_yaw,

            "mean_abs_yaw_rate_deg_s":
                float(
                    np.rad2deg(
                        mean_abs_yaw
                    )
                ),

            "mean_upright":
                float(
                    np.mean(
                        upright_values
                    )
                ),

            "minimum_upright":
                float(
                    np.min(
                        upright_values
                    )
                ),

            "minimum_height":
                float(
                    np.min(
                        height_values
                    )
                ),

            "mean_action_delta_mse":
                float(
                    np.mean(
                        action_delta_values
                    )
                ),

            "net_displacement":
                displacement,

            "path_length":
                path_length,

            "path_efficiency":
                path_efficiency,

            "terminated":
                int(
                    terminated
                ),

            "truncated":
                int(
                    truncated
                ),
        }

        episode_rows.append(
            episode_row
        )

        print()
        print(
            f"Episode {episode:02d}"
        )

        print(
            f"  reward          : "
            f"{total_reward:+.3f}"
        )

        print(
            f"  forward         : "
            f"{mean_forward:+.5f} m/s"
        )

        print(
            f"  |forward|       : "
            f"{mean_abs_forward:.5f} m/s"
        )

        print(
            f"  |lateral|       : "
            f"{mean_abs_lateral:.5f} m/s"
        )

        print(
            f"  directional     : "
            f"{directional_ratio:.4f}"
        )

        print(
            f"  |yaw|           : "
            f"{np.rad2deg(mean_abs_yaw):.2f} deg/s"
        )

        print(
            f"  displacement    : "
            f"{displacement:.4f} m"
        )

        print(
            f"  path length     : "
            f"{path_length:.4f} m"
        )

        print(
            f"  path efficiency : "
            f"{path_efficiency:.4f}"
        )

        print(
            f"  min upright     : "
            f"{np.min(upright_values):.4f}"
        )

        print(
            f"  terminated      : "
            f"{terminated}"
        )

    env.close()

    # ========================================================
    # SAVE
    # ========================================================

    step_path = (
        RESULT_ROOT
        / "final_eval_steps.csv"
    )

    episode_path = (
        RESULT_ROOT
        / "final_eval_episodes.csv"
    )

    save_csv(
        step_path,
        step_rows,
    )

    save_csv(
        episode_path,
        episode_rows,
    )

    # ========================================================
    # OVERALL
    # ========================================================

    rewards = np.asarray([
        r["total_reward"]
        for r in episode_rows
    ])

    forward = np.asarray([
        r["mean_forward_velocity"]
        for r in episode_rows
    ])

    lateral = np.asarray([
        r["mean_abs_lateral_velocity"]
        for r in episode_rows
    ])

    yaw = np.asarray([
        r["mean_abs_yaw_rate_deg_s"]
        for r in episode_rows
    ])

    directional = np.asarray([
        r["directional_ratio"]
        for r in episode_rows
    ])

    path_efficiency = np.asarray([
        r["path_efficiency"]
        for r in episode_rows
    ])

    falls = sum(
        r["terminated"]
        for r in episode_rows
    )

    print()
    print("=" * 70)

    print(
        "003-01 PPO FINAL SUMMARY"
    )

    print("=" * 70)

    print(
        f"Episodes                    : "
        f"{len(episode_rows)}"
    )

    print(
        f"Mean episode reward         : "
        f"{np.mean(rewards):+.3f}"
    )

    print(
        f"Mean forward velocity       : "
        f"{np.mean(forward):+.5f} m/s"
    )

    print(
        f"Mean |lateral velocity|     : "
        f"{np.mean(lateral):.5f} m/s"
    )

    print(
        f"Mean directional ratio      : "
        f"{np.mean(directional):.4f}"
    )

    print(
        f"Mean |yaw rate|             : "
        f"{np.mean(yaw):.3f} deg/s"
    )

    print(
        f"Mean path efficiency        : "
        f"{np.mean(path_efficiency):.4f}"
    )

    print(
        f"Falls / terminations        : "
        f"{falls}/{len(episode_rows)}"
    )

    print("=" * 70)

    return episode_rows


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "003-01 Sesame PPO baseline"
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQ,
    )

    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--render-final",
        action="store_true",
    )

    parser.add_argument(
        "--skip-env-check",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 72)

    print(
        "01 / 003-01 SESAME PPO BASELINE"
    )

    print("=" * 72)

    print(
        f"Total timesteps   : "
        f"{args.timesteps}"
    )

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print(
        f"Device requested  : "
        f"{args.device}"
    )

    print()
    print(
        "--- PPO ---"
    )

    print(
        f"learning_rate     : "
        f"{LEARNING_RATE}"
    )

    print(
        f"n_steps           : "
        f"{N_STEPS}"
    )

    print(
        f"batch_size        : "
        f"{BATCH_SIZE}"
    )

    print(
        f"n_epochs          : "
        f"{N_EPOCHS}"
    )

    print(
        f"gamma             : "
        f"{GAMMA}"
    )

    print(
        f"gae_lambda        : "
        f"{GAE_LAMBDA}"
    )

    print(
        f"clip_range        : "
        f"{CLIP_RANGE}"
    )

    print(
        f"ent_coef          : "
        f"{ENT_COEF}"
    )

    print()
    print(
        "--- REWARD V1 ---"
    )

    print(
        f"forward           : "
        f"{W_FORWARD}"
    )

    print(
        f"lateral penalty   : "
        f"{W_LATERAL}"
    )

    print(
        f"yaw penalty       : "
        f"{W_YAW}"
    )

    print(
        f"upright penalty   : "
        f"{W_UPRIGHT}"
    )

    print(
        f"action-rate       : "
        f"{W_ACTION_RATE}"
    )

    print(
        f"alive             : "
        f"{ALIVE_REWARD}"
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    save_experiment_config(
        args
    )

    # ========================================================
    # CHECK ENV
    # ========================================================

    if not args.skip_env_check:

        validate_environment()

    # ========================================================
    # VECTOR ENV
    # ========================================================

    train_env = DummyVecEnv(
        [
            make_training_env(
                args.seed
            )
        ]
    )

    eval_env = DummyVecEnv(
        [
            make_eval_env(
                args.seed + 10_000
            )
        ]
    )

    # ========================================================
    # CALLBACKS
    # ========================================================

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=(
                args.checkpoint_freq
            ),
            save_path=str(
                CHECKPOINT_DIR
            ),
            name_prefix=(
                "sesame_ppo"
            ),
            verbose=2,
        )
    )

    eval_callback = (
        EvalCallback(
            eval_env,
            best_model_save_path=str(
                BEST_MODEL_DIR
            ),
            log_path=str(
                EVAL_DIR
            ),
            eval_freq=(
                args.eval_freq
            ),
            n_eval_episodes=(
                EVAL_EPISODES
            ),
            deterministic=True,
            render=False,
            verbose=1,
        )
    )

    callbacks = CallbackList(
        [
            checkpoint_callback,
            eval_callback,
        ]
    )

    # ========================================================
    # TENSORBOARD
    # ========================================================

    tensorboard_log = None

    try:

        import tensorboard  # noqa: F401

        tensorboard_log = str(
            TENSORBOARD_DIR
        )

        print(
            "TensorBoard       : ENABLED"
        )

    except ImportError:

        print(
            "TensorBoard       : DISABLED "
            "(tensorboard not installed)"
        )

    # ========================================================
    # PPO
    # ========================================================

    print()
    print(
        "=== CREATE PPO MODEL ==="
    )

    model = PPO(

        policy="MlpPolicy",

        env=train_env,

        learning_rate=(
            LEARNING_RATE
        ),

        n_steps=(
            N_STEPS
        ),

        batch_size=(
            BATCH_SIZE
        ),

        n_epochs=(
            N_EPOCHS
        ),

        gamma=(
            GAMMA
        ),

        gae_lambda=(
            GAE_LAMBDA
        ),

        clip_range=(
            CLIP_RANGE
        ),

        ent_coef=(
            ENT_COEF
        ),

        vf_coef=(
            VF_COEF
        ),

        max_grad_norm=(
            MAX_GRAD_NORM
        ),

        normalize_advantage=True,

        tensorboard_log=(
            tensorboard_log
        ),

        seed=(
            args.seed
        ),

        device=(
            args.device
        ),

        verbose=1,
    )

    print(
        f"Actual device     : "
        f"{model.device}"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 72)

    print(
        "START PPO TRAINING"
    )

    print("=" * 72)

    model.learn(

        total_timesteps=(
            args.timesteps
        ),

        callback=(
            callbacks
        ),

        log_interval=1,

        tb_log_name=(
            "003_01_ppo"
        ),

        progress_bar=False,
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model_path = (
        MODEL_DIR
        / "sesame_ppo_final"
    )

    model.save(
        str(
            final_model_path
        )
    )

    print()
    print(
        f"Final model saved : "
        f"{final_model_path}.zip"
    )

    # ========================================================
    # SELECT MODEL FOR FINAL EVALUATION
    # ========================================================

    best_model_path = (
        BEST_MODEL_DIR
        / "best_model.zip"
    )

    if best_model_path.exists():

        print(
            "Final evaluation  : "
            "BEST EVAL MODEL"
        )

        evaluation_model = (
            PPO.load(
                str(
                    best_model_path
                ),
                device=args.device,
            )
        )

    else:

        print(
            "Final evaluation  : "
            "FINAL TRAINING MODEL"
        )

        evaluation_model = (
            model
        )

    # ========================================================
    # DETAILED EVALUATION
    # ========================================================

    evaluate_policy_detailed(

        evaluation_model,

        episodes=(
            args.final_eval_episodes
        ),

        seed=(
            args.seed
            + 20_000
        ),

        render=(
            args.render_final
        ),
    )

    # ========================================================
    # CLOSE
    # ========================================================

    train_env.close()

    eval_env.close()

    print()
    print(
        "003-01 COMPLETE"
    )

    print(
        f"Results:"
    )

    print(
        RESULT_ROOT
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()

"""

