#!/usr/bin/env python3

"""
003-02 — Reward V2 Yaw-Penalty Ablation
=======================================

Research question:
Can stronger yaw regularization reduce rotation
while preserving forward locomotion?

Only experimental change:

    003-01 Reward V1:
        W_YAW = 0.05

    003-02 Reward V2:
        W_YAW = 0.20

Everything else remains unchanged.
"""

from pathlib import Path

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import (
    EvalCallback,
)

from stable_baselines3.common.env_checker import (
    check_env,
)

from stable_baselines3.common.monitor import (
    Monitor,
)

from sesame_rl_env import (
    SesameRLEnv,
    CONTROL_DT,
    EPISODE_STEPS,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

SEED = 1000

TOTAL_TIMESTEPS = 300_000

W_YAW_V1 = 0.05
W_YAW_V2 = 0.20


# ============================================================
# PPO CONFIG
#
# Same as 003-01
# ============================================================

LEARNING_RATE = 3e-4

N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_RANGE = 0.20

ENT_COEF = 0.01


# ============================================================
# EVALUATION CONFIG
# ============================================================

EVAL_FREQ = 10_000

EVAL_EPISODES = 5

FINAL_EVAL_EPISODES = 5


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

RESULT_DIR = (
    PROJECT_ROOT
    / "003_rl_results"
    / "02_reward_v2_yaw"
)

BEST_MODEL_DIR = (
    RESULT_DIR
    / "best_model"
)

EVAL_DIR = (
    RESULT_DIR
    / "eval"
)

MODEL_DIR = (
    RESULT_DIR
    / "models"
)

MONITOR_DIR = (
    RESULT_DIR
    / "monitor"
)


for directory in [
    RESULT_DIR,
    BEST_MODEL_DIR,
    EVAL_DIR,
    MODEL_DIR,
    MONITOR_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print(
    "02 / 003-02 REWARD V2 — YAW PENALTY ABLATION"
)
print("=" * 70)

print(
    f"W_YAW V1          : {W_YAW_V1}"
)

print(
    f"W_YAW V2          : {W_YAW_V2}"
)

print(
    f"Total timesteps   : {TOTAL_TIMESTEPS}"
)

print(
    f"Eval frequency    : {EVAL_FREQ}"
)

print(
    f"Eval episodes     : {EVAL_EPISODES}"
)

print(
    f"Seed              : {SEED}"
)

print("=" * 70)


# ============================================================
# 1. ENVIRONMENT CHECK
# ============================================================

print()
print(
    "=== ENVIRONMENT CHECK ==="
)


test_env = SesameRLEnv(
    w_yaw=W_YAW_V2
)

check_env(
    test_env,
    warn=True,
)

obs, info = test_env.reset(
    seed=SEED
)

print(
    f"Observation shape : {obs.shape}"
)

print(
    f"Action shape      : {test_env.action_space.shape}"
)

print(
    f"W_YAW             : {W_YAW_V2}"
)

print(
    "SB3 check_env     : PASS"
)

test_env.close()


# ============================================================
# 2. TRAINING ENVIRONMENT
# ============================================================

train_env = Monitor(

    SesameRLEnv(
        w_yaw=W_YAW_V2
    ),

    filename=str(
        MONITOR_DIR
        / "train"
    ),
)

train_env.reset(
    seed=SEED
)


# ============================================================
# 3. EVALUATION ENVIRONMENT
#
# IMPORTANT:
# Independent env used to select best checkpoint.
# ============================================================

eval_env = Monitor(

    SesameRLEnv(
        w_yaw=W_YAW_V2
    ),

    filename=str(
        MONITOR_DIR
        / "eval"
    ),
)

eval_env.reset(
    seed=SEED + 10_000
)


# ============================================================
# 4. BEST MODEL CALLBACK
# ============================================================

eval_callback = EvalCallback(

    eval_env,

    best_model_save_path=str(
        BEST_MODEL_DIR
    ),

    log_path=str(
        EVAL_DIR
    ),

    eval_freq=EVAL_FREQ,

    n_eval_episodes=EVAL_EPISODES,

    deterministic=True,

    render=False,

    verbose=1,
)


# ============================================================
# 5. PPO
# ============================================================

print()
print(
    "=== CREATE PPO MODEL ==="
)


model = PPO(

    "MlpPolicy",

    train_env,

    learning_rate=LEARNING_RATE,

    n_steps=N_STEPS,

    batch_size=BATCH_SIZE,

    n_epochs=N_EPOCHS,

    gamma=GAMMA,

    gae_lambda=GAE_LAMBDA,

    clip_range=CLIP_RANGE,

    ent_coef=ENT_COEF,

    seed=SEED,

    verbose=1,

    device="auto",
)


print(
    f"Device            : {model.device}"
)


# ============================================================
# 6. TRAIN
# ============================================================

print()
print("=" * 70)
print(
    "START 003-02 PPO TRAINING"
)
print("=" * 70)


model.learn(

    total_timesteps=TOTAL_TIMESTEPS,

    callback=eval_callback,

    log_interval=1,
)


# ============================================================
# 7. SAVE FINAL MODEL
#
# We save it for analysis, but DO NOT automatically
# treat it as the experimental result.
# ============================================================

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "ppo_reward_v2_final"
)

model.save(
    str(
        FINAL_MODEL_PATH
    )
)


print()
print(
    f"Final model saved : "
    f"{FINAL_MODEL_PATH}.zip"
)


# ============================================================
# 8. LOAD BEST MODEL
# ============================================================

BEST_MODEL_PATH = (
    BEST_MODEL_DIR
    / "best_model.zip"
)


if not BEST_MODEL_PATH.exists():

    raise FileNotFoundError(
        "Best model was not created:\n"
        f"{BEST_MODEL_PATH}"
    )


print(
    f"Best model        : "
    f"{BEST_MODEL_PATH}"
)


best_model = PPO.load(
    str(
        BEST_MODEL_PATH
    )
)


# ============================================================
# 9. FINAL DETAILED EVALUATION
# ============================================================

print()
print(
    "=== 003-02 BEST MODEL EVALUATION ==="
)


final_env = SesameRLEnv(
    w_yaw=W_YAW_V2
)


results = []


for episode in range(
    FINAL_EVAL_EPISODES
):

    obs, _ = final_env.reset(
        seed=(
            20_000
            + episode
        )
    )

    total_reward = 0.0

    forward_values = []
    abs_forward_values = []
    lateral_values = []
    yaw_values = []
    upright_values = []

    # --------------------------------------------------------
    # Start position
    # --------------------------------------------------------

    start_position = (
        final_env.data.xpos[
            final_env.base_body_id
        ].copy()
    )

    previous_position = (
        start_position.copy()
    )

    path_length = 0.0

    terminated = False
    truncated = False


    # ========================================================
    # ROLLOUT
    # ========================================================

    for step in range(
        EPISODE_STEPS
    ):

        action, _ = (
            best_model.predict(
                obs,
                deterministic=True,
            )
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = final_env.step(
            action
        )

        total_reward += (
            reward
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

        forward_values.append(
            forward
        )

        abs_forward_values.append(
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


        # ----------------------------------------------------
        # Path length
        # ----------------------------------------------------

        current_position = (
            final_env.data.xpos[
                final_env.base_body_id
            ].copy()
        )

        path_length += float(

            np.linalg.norm(

                current_position[:2]
                - previous_position[:2]
            )
        )

        previous_position = (
            current_position.copy()
        )


        if (
            terminated
            or truncated
        ):

            break


    # ========================================================
    # EPISODE METRICS
    # ========================================================

    end_position = (
        final_env.data.xpos[
            final_env.base_body_id
        ].copy()
    )


    displacement = float(

        np.linalg.norm(

            end_position[:2]
            - start_position[:2]
        )
    )


    mean_forward = float(
        np.mean(
            forward_values
        )
    )


    mean_abs_forward = float(
        np.mean(
            abs_forward_values
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

        / (
            mean_abs_forward
            + mean_abs_lateral
            + 1e-9
        )
    )


    if path_length > 1e-9:

        path_efficiency = (
            displacement
            / path_length
        )

    else:

        path_efficiency = 0.0


    min_upright = float(
        np.min(
            upright_values
        )
    )


    result = {

        "reward":
            total_reward,

        "forward":
            mean_forward,

        "abs_forward":
            mean_abs_forward,

        "abs_lateral":
            mean_abs_lateral,

        "directional":
            directional_ratio,

        "yaw_deg":
            float(
                np.rad2deg(
                    mean_abs_yaw
                )
            ),

        "displacement":
            displacement,

        "path_length":
            path_length,

        "path_efficiency":
            path_efficiency,

        "min_upright":
            min_upright,

        "terminated":
            terminated,
    }


    results.append(
        result
    )


    # ========================================================
    # PRINT EPISODE
    # ========================================================

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
        f"{min_upright:.4f}"
    )

    print(
        f"  terminated      : "
        f"{terminated}"
    )


# ============================================================
# 10. CLOSE ENVIRONMENTS
# ============================================================

train_env.close()

eval_env.close()

final_env.close()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

mean_reward = float(
    np.mean([
        r["reward"]
        for r in results
    ])
)


mean_forward = float(
    np.mean([
        r["forward"]
        for r in results
    ])
)


mean_abs_forward = float(
    np.mean([
        r["abs_forward"]
        for r in results
    ])
)


mean_lateral = float(
    np.mean([
        r["abs_lateral"]
        for r in results
    ])
)


mean_directional = float(
    np.mean([
        r["directional"]
        for r in results
    ])
)


mean_yaw = float(
    np.mean([
        r["yaw_deg"]
        for r in results
    ])
)


mean_path_efficiency = float(
    np.mean([
        r["path_efficiency"]
        for r in results
    ])
)


falls = int(
    sum(
        r["terminated"]
        for r in results
    )
)


print()
print("=" * 70)
print(
    "003-02 REWARD V2 BEST MODEL SUMMARY"
)
print("=" * 70)

print(
    f"W_YAW                      : "
    f"{W_YAW_V2}"
)

print(
    f"Episodes                   : "
    f"{len(results)}"
)

print(
    f"Mean episode reward        : "
    f"{mean_reward:+.3f}"
)

print(
    f"Mean forward velocity      : "
    f"{mean_forward:+.5f} m/s"
)

print(
    f"Mean |forward velocity|    : "
    f"{mean_abs_forward:.5f} m/s"
)

print(
    f"Mean |lateral velocity|    : "
    f"{mean_lateral:.5f} m/s"
)

print(
    f"Mean directional ratio     : "
    f"{mean_directional:.4f}"
)

print(
    f"Mean |yaw rate|            : "
    f"{mean_yaw:.3f} deg/s"
)

print(
    f"Mean path efficiency       : "
    f"{mean_path_efficiency:.4f}"
)

print(
    f"Falls / terminations       : "
    f"{falls}/{len(results)}"
)

print()
print(
    f"Best model:"
)

print(
    BEST_MODEL_PATH
)

print()
print(
    f"Final model:"
)

print(
    f"{FINAL_MODEL_PATH}.zip"
)

print("=" * 70)


# ============================================================
# 12. V1 REFERENCE
# ============================================================

print()
print(
    "003-01 REWARD V1 REFERENCE"
)

print(
    "Forward velocity           : "
    "+0.20151 m/s"
)

print(
    "|Lateral velocity|         : "
    "0.07469 m/s"
)

print(
    "Directional ratio          : "
    "0.7360"
)

print(
    "|Yaw rate|                 : "
    "127.396 deg/s"
)

print(
    "Path efficiency            : "
    "0.4531"
)

print()
print(
    "Compare V2 against V1 using motion metrics, "
    "NOT raw reward."
)

print()
print(
    "003-02 COMPLETE"
)