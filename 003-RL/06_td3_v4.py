#!/usr/bin/env python3

"""
003-06 / V4 — TD3 Baseline
==========================

Comparison:
V2 = PPO + Reward V2
V3 = SAC + Reward V2
V4 = TD3 + Reward V2

Fixed:
- Same MuJoCo environment
- Observation = 33D
- Action = 8D
- W_YAW = 0.20
- 300k environment steps
- Seed = 1000

Changed:
- RL algorithm -> TD3
"""

from pathlib import Path
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from sesame_rl_env import SesameRLEnv


# ============================================================
# EXPERIMENT
# ============================================================

SEED = 1000
TOTAL_TIMESTEPS = 300_000

W_YAW = 0.20

# 如果 V3 是 CPU，V4 也先用 CPU，方便比較
DEVICE = "cpu"


# ============================================================
# TD3 PARAMETERS
# ============================================================

LEARNING_RATE = 3e-4

BUFFER_SIZE = 300_000
LEARNING_STARTS = 10_000

BATCH_SIZE = 256

GAMMA = 0.99
TAU = 0.005

TRAIN_FREQ = 1
GRADIENT_STEPS = 1

POLICY_DELAY = 2

TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5


# ============================================================
# EXPLORATION NOISE
# ============================================================

N_ACTIONS = 8

ACTION_NOISE = NormalActionNoise(
    mean=np.zeros(N_ACTIONS),
    sigma=0.10 * np.ones(N_ACTIONS),
)


# ============================================================
# EVALUATION
# ============================================================

EVAL_FREQ = 10_000
EVAL_EPISODES = 5

CHECKPOINT_FREQ = 50_000


# ============================================================
# PATHS
# ============================================================

RESULT_DIR = Path(
    "003_rl_results/06_td3_v4"
)

MODEL_DIR = RESULT_DIR / "models"
BEST_DIR = RESULT_DIR / "best_model"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
EVAL_DIR = RESULT_DIR / "eval"
MONITOR_DIR = RESULT_DIR / "monitor"

for path in [
    RESULT_DIR,
    MODEL_DIR,
    BEST_DIR,
    CHECKPOINT_DIR,
    EVAL_DIR,
    MONITOR_DIR,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# ENVIRONMENTS
# ============================================================

train_env = Monitor(
    SesameRLEnv(
        w_yaw=W_YAW
    ),
    filename=str(
        MONITOR_DIR / "train"
    ),
)

eval_env = Monitor(
    SesameRLEnv(
        w_yaw=W_YAW
    ),
    filename=str(
        MONITOR_DIR / "eval"
    ),
)

train_env.reset(
    seed=SEED
)

eval_env.reset(
    seed=SEED + 10_000
)


# ============================================================
# CALLBACKS
# ============================================================

eval_callback = EvalCallback(
    eval_env,

    best_model_save_path=str(
        BEST_DIR
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


checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,

    save_path=str(
        CHECKPOINT_DIR
    ),

    name_prefix="td3_v4",

    verbose=1,
)


callbacks = CallbackList(
    [
        eval_callback,
        checkpoint_callback,
    ]
)


# ============================================================
# TD3 MODEL
# ============================================================

print()
print("=" * 60)
print("003-06 / V4 — TD3")
print("=" * 60)

print("Algorithm        : TD3")
print("Reward           : V2")
print("W_YAW            :", W_YAW)
print("Timesteps        :", TOTAL_TIMESTEPS)
print("Seed             :", SEED)
print("Device           :", DEVICE)

print("Learning rate    :", LEARNING_RATE)
print("Buffer size      :", BUFFER_SIZE)
print("Learning starts  :", LEARNING_STARTS)
print("Batch size       :", BATCH_SIZE)

print("Policy delay     :", POLICY_DELAY)
print("Target noise     :", TARGET_POLICY_NOISE)
print("Noise clip       :", TARGET_NOISE_CLIP)
print("Action noise SD  : 0.10")

print("=" * 60)


model = TD3(
    "MlpPolicy",

    train_env,

    learning_rate=LEARNING_RATE,

    buffer_size=BUFFER_SIZE,

    learning_starts=LEARNING_STARTS,

    batch_size=BATCH_SIZE,

    tau=TAU,

    gamma=GAMMA,

    train_freq=TRAIN_FREQ,

    gradient_steps=GRADIENT_STEPS,

    action_noise=ACTION_NOISE,

    policy_delay=POLICY_DELAY,

    target_policy_noise=TARGET_POLICY_NOISE,

    target_noise_clip=TARGET_NOISE_CLIP,

    seed=SEED,

    device=DEVICE,

    verbose=1,
)


# ============================================================
# TRAIN
# ============================================================

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,

    callback=callbacks,

    log_interval=10,
)


# ============================================================
# SAVE FINAL
# ============================================================

FINAL_PATH = (
    MODEL_DIR
    / "td3_v4_final"
)

model.save(
    str(FINAL_PATH)
)


train_env.close()
eval_env.close()


print()
print("=" * 60)
print("V4 TD3 COMPLETE")
print("=" * 60)

print(
    "Best model :",
    BEST_DIR / "best_model.zip"
)

print(
    "Final model:",
    str(FINAL_PATH) + ".zip"
)