#!/usr/bin/env python3

"""
003-05 / V3 — SAC Baseline
==========================

Goal:
Compare SAC against PPO under the SAME:

- MuJoCo model
- 33D observation
- 8D continuous action
- Reward V2
- episode length
- physics / servo dynamics

Only the RL algorithm changes:

    PPO -> SAC
"""

from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from sesame_rl_env import SesameRLEnv


# ============================================================
# EXPERIMENT
# ============================================================

SEED = 1000

TOTAL_TIMESTEPS = 300_000

# Keep Reward V2 fixed
W_YAW = 0.20


# ============================================================
# SAC PARAMETERS
# ============================================================

LEARNING_RATE = 3e-4

BUFFER_SIZE = 300_000

LEARNING_STARTS = 10_000

BATCH_SIZE = 256

GAMMA = 0.99

TAU = 0.005

TRAIN_FREQ = 1

GRADIENT_STEPS = 1

ENT_COEF = "auto"


# ============================================================
# EVALUATION
# ============================================================

EVAL_FREQ = 10_000

EVAL_EPISODES = 5


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

RESULT_DIR = (
    PROJECT_ROOT
    / "003_rl_results"
    / "05_sac_v3"
)

BEST_MODEL_DIR = (
    RESULT_DIR
    / "best_model"
)

MODEL_DIR = (
    RESULT_DIR
    / "models"
)

EVAL_DIR = (
    RESULT_DIR
    / "eval"
)

MONITOR_DIR = (
    RESULT_DIR
    / "monitor"
)


for directory in [
    RESULT_DIR,
    BEST_MODEL_DIR,
    MODEL_DIR,
    EVAL_DIR,
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
print("=" * 68)
print("003-05 / V3 — SAC BASELINE")
print("=" * 68)

print(f"Algorithm          : SAC")
print(f"Reward W_YAW       : {W_YAW}")
print(f"Total timesteps    : {TOTAL_TIMESTEPS}")
print(f"Replay buffer      : {BUFFER_SIZE}")
print(f"Learning starts    : {LEARNING_STARTS}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Gamma              : {GAMMA}")
print(f"Tau                : {TAU}")
print(f"Entropy coefficient: {ENT_COEF}")
print(f"Seed               : {SEED}")

print("=" * 68)


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

print()
print("=== ENVIRONMENT CHECK ===")

test_env = SesameRLEnv(
    w_yaw=W_YAW
)

check_env(
    test_env,
    warn=True,
)

obs, _ = test_env.reset(
    seed=SEED
)

print(
    f"Observation shape : "
    f"{obs.shape}"
)

print(
    f"Action shape      : "
    f"{test_env.action_space.shape}"
)

print(
    "SB3 check_env     : PASS"
)

test_env.close()


# ============================================================
# TRAINING ENVIRONMENT
# ============================================================

train_env = Monitor(

    SesameRLEnv(
        w_yaw=W_YAW
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
# EVALUATION ENVIRONMENT
# ============================================================

eval_env = Monitor(

    SesameRLEnv(
        w_yaw=W_YAW
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
# SAVE BEST MODEL
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
# SAC
# ============================================================

print()
print("=== CREATE SAC MODEL ===")


model = SAC(

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

    ent_coef=ENT_COEF,

    seed=SEED,

    verbose=1,

    device="cuda",
)


print(
    f"Device             : "
    f"{model.device}"
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 68)
print("START SAC TRAINING")
print("=" * 68)


model.learn(

    total_timesteps=TOTAL_TIMESTEPS,

    callback=eval_callback,

    log_interval=10,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "sac_v3_final"
)

model.save(
    str(
        FINAL_MODEL_PATH
    )
)


# ============================================================
# CLOSE
# ============================================================

train_env.close()

eval_env.close()


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 68)
print("003-05 / V3 SAC TRAINING COMPLETE")
print("=" * 68)

print(
    f"Best model : "
    f"{BEST_MODEL_DIR / 'best_model.zip'}"
)

print(
    f"Final model: "
    f"{FINAL_MODEL_PATH}.zip"
)

print()
print(
    "Use best_model.zip for formal evaluation/comparison."
)

print("=" * 68)