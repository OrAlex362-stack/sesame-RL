import time

import numpy as np

from stable_baselines3 import PPO

from sesame_rl_env import (
    SesameRLEnv,
    CONTROL_DT,
)


MODEL_PATH = (
    "003_rl_results/"
    "02_reward_v2_yaw/"
    "ppo_reward_v2.zip"
)

"""
MODEL_PATH = (
    "003_rl_results/"
    "01_ppo_baseline_001/"
    "models/"
    "sesame_ppo_final.zip"
)

MODEL_PATH = (
    "003_rl_results/"
    "02_reward_v2_yaw/"
    "ppo_reward_v2.zip"
)

MODEL_PATH = (
    "003_rl_results/"
    "01_ppo_baseline_001/"
    "best_models/"
    "best_model"
)

"""

# ============================================================
# ENV WITH MUJOCO VIEWER
# ============================================================

env = SesameRLEnv(
    render_mode="human",
    w_yaw=0.20,
)


# ============================================================
# LOAD PPO
# ============================================================

model = PPO.load(
    MODEL_PATH
)


# ============================================================
# RESET
# ============================================================

obs, info = env.reset(
    seed=2000
)


# ============================================================
# RUN POLICY
# ============================================================

for step in range(1000):

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

    print(
        f"\r"
        f"step={step:4d} | "
        f"forward={info['forward_velocity']:+.3f} m/s | "
        f"lateral={info['lateral_velocity']:+.3f} m/s | "
        f"yaw={info['yaw_rate']:+.3f} rad/s",
        end=""
    )

    # Make viewer approximately real-time
    time.sleep(
        CONTROL_DT
    )

    if terminated:
        print("\nRobot terminated.")
        break

    if truncated:
        print("\nEpisode complete.")
        break


env.close()