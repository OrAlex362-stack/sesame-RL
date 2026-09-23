#!/usr/bin/env python3

"""
003-03 — Generic MuJoCo Policy Viewer
=====================================

Supports:
- PPO
- SAC
- TD3

Examples:
V2 = PPO + Reward V2
V3 = SAC + Reward V2
V4 = TD3 + Reward V2

Usage:
    python 003-RL/03_view_ppo.py

Only change:
    ALGO
    MODEL_PATH
    W_YAW
"""

from pathlib import Path
import time

from stable_baselines3 import PPO, SAC, TD3

from sesame_rl_env import SesameRLEnv


# ============================================================
# SELECT MODEL
# ============================================================

# Choose:
# "PPO"
# "SAC"
# "TD3"

ALGO = "TD3"


# ------------------------------------------------------------
# V2 PPO example
# ------------------------------------------------------------

# ALGO = "PPO"
#
# MODEL_PATH = (
#     "003_rl_results/"
#     "02_reward_v2_yaw/"
#     "best_model/"
#     "best_model.zip"
# )


# ------------------------------------------------------------
# V3 SAC
# ------------------------------------------------------------

MODEL_PATH = (
    "003_rl_results/"
    "06_td3_v4/"
    "best_model/"
    "best_model.zip"
)


# ------------------------------------------------------------
# V4 TD3 example
# ------------------------------------------------------------

# ALGO = "TD3"
#
# MODEL_PATH = (
#     "003_rl_results/"
#     "06_td3_v4/"
#     "best_model/"
#     "best_model.zip"
# )


# ============================================================
# ENVIRONMENT
# ============================================================

# V2 / V3 / V4 use Reward V2
W_YAW = 0.20

SEED = 2000

MAX_STEPS = 1000

CONTROL_DT = 0.020

DETERMINISTIC = True


# ============================================================
# ALGORITHM LOADER
# ============================================================

ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


# ============================================================
# CHECK MODEL
# ============================================================

model_path = Path(MODEL_PATH)

if not model_path.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n"
        f"{model_path.resolve()}\n"
    )


if ALGO not in ALGORITHMS:

    raise ValueError(
        f"Unknown ALGO: {ALGO}\n"
        f"Choose from: {list(ALGORITHMS.keys())}"
    )


# ============================================================
# CREATE ENV
# ============================================================

env = SesameRLEnv(
    render_mode="human",
    w_yaw=W_YAW,
)


# ============================================================
# LOAD MODEL
# ============================================================

ModelClass = ALGORITHMS[ALGO]

print()
print("=" * 65)
print("003-03 — POLICY VIEWER")
print("=" * 65)

print(
    f"Algorithm      : {ALGO}"
)

print(
    f"Model          : {model_path}"
)

print(
    f"W_YAW          : {W_YAW}"
)

print(
    f"Seed           : {SEED}"
)

print(
    f"Deterministic  : {DETERMINISTIC}"
)

print("=" * 65)
print()


model = ModelClass.load(
    str(model_path),
    env=env,
    device="auto",
)


print(
    f"Model device   : {model.device}"
)

print()
print("Viewer starting...")
print()


# ============================================================
# RESET
# ============================================================

obs, info = env.reset(
    seed=SEED
)


# ============================================================
# METRICS
# ============================================================

total_reward = 0.0

forward_sum = 0.0

abs_forward_sum = 0.0

abs_lateral_sum = 0.0

abs_yaw_sum = 0.0

steps_run = 0


# ============================================================
# VIEW LOOP
# ============================================================

try:

    for step in range(MAX_STEPS):

        # ----------------------------------------------------
        # POLICY
        # ----------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=DETERMINISTIC,
        )


        # ----------------------------------------------------
        # ENV STEP
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )


        # ----------------------------------------------------
        # READ METRICS
        # ----------------------------------------------------

        forward = info.get(
            "forward_velocity",
            0.0
        )

        lateral = info.get(
            "lateral_velocity",
            0.0
        )

        yaw_rate = info.get(
            "yaw_rate",
            0.0
        )

        upright = info.get(
            "upright",
            0.0
        )

        height = info.get(
            "body_height",
            0.0
        )


        # ----------------------------------------------------
        # ACCUMULATE
        # ----------------------------------------------------

        total_reward += reward

        forward_sum += forward

        abs_forward_sum += abs(
            forward
        )

        abs_lateral_sum += abs(
            lateral
        )

        abs_yaw_sum += abs(
            yaw_rate
        )

        steps_run += 1


        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        if (
            step % 10 == 0
            or terminated
            or truncated
        ):

            print(
                f"step={step:4d} | "
                f"reward={reward:+7.3f} | "
                f"fwd={forward:+7.3f} | "
                f"lat={lateral:+7.3f} | "
                f"yaw={yaw_rate:+7.3f} | "
                f"upright={upright:6.3f} | "
                f"z={height:6.3f}"
            )


        # ----------------------------------------------------
        # REAL-TIME VIEWING
        # ----------------------------------------------------

        time.sleep(
            CONTROL_DT
        )


        # ----------------------------------------------------
        # END EPISODE
        # ----------------------------------------------------

        if terminated:

            print()
            print(
                f"Episode TERMINATED "
                f"at step {step}"
            )

            break


        if truncated:

            print()
            print(
                f"Episode TRUNCATED "
                f"at step {step}"
            )

            break


except KeyboardInterrupt:

    print()
    print(
        "Viewer stopped manually."
    )


finally:

    env.close()


# ============================================================
# SUMMARY
# ============================================================

if steps_run > 0:

    mean_forward = (
        forward_sum
        / steps_run
    )

    mean_abs_forward = (
        abs_forward_sum
        / steps_run
    )

    mean_abs_lateral = (
        abs_lateral_sum
        / steps_run
    )

    mean_abs_yaw = (
        abs_yaw_sum
        / steps_run
    )

    directional_ratio = (
        mean_abs_forward
        /
        (
            mean_abs_forward
            + mean_abs_lateral
            + 1e-8
        )
    )

else:

    mean_forward = 0.0
    mean_abs_forward = 0.0
    mean_abs_lateral = 0.0
    mean_abs_yaw = 0.0
    directional_ratio = 0.0


print()
print("=" * 65)
print("VIEWER SUMMARY")
print("=" * 65)

print(
    f"Algorithm              : {ALGO}"
)

print(
    f"Steps                  : {steps_run}"
)

print(
    f"Total reward           : {total_reward:+.3f}"
)

print(
    f"Mean forward velocity  : {mean_forward:+.5f} m/s"
)

print(
    f"Mean |forward|         : {mean_abs_forward:.5f} m/s"
)

print(
    f"Mean |lateral|         : {mean_abs_lateral:.5f} m/s"
)

print(
    f"Mean |yaw rate|        : {mean_abs_yaw:.5f} rad/s"
)

print(
    f"Directional ratio      : {directional_ratio:.4f}"
)

print("=" * 65)