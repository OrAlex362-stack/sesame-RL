#!/usr/bin/env python3

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "td3"
TOTAL_TIMESTEPS = 500_000
SEED = 42


def load_env_class():
    spec = importlib.util.spec_from_file_location("command_env", ROOT / "01-command-env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CommandEnv


def write_training_summary(monitor_file, output_file):
    with monitor_file.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    timestep = 0
    with output_file.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["episode", "timestep", "episode_reward", "episode_length", "elapsed_seconds"])
        for episode, row in enumerate(rows, 1):
            timestep += int(row["l"])
            writer.writerow([episode, timestep, row["r"], row["l"], row["t"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a 5,000-step pipeline check")
    args = parser.parse_args()
    total_steps = 5_000 if args.smoke else TOTAL_TIMESTEPS
    output_dir = RESULT_DIR / "smoke_test" if args.smoke else RESULT_DIR
    eval_freq = 1_000 if args.smoke else 10_000
    checkpoint_freq = 2_500 if args.smoke else 50_000

    best_dir = output_dir / "best_model"
    checkpoint_dir = output_dir / "checkpoints"
    monitor_dir = output_dir / "monitor"
    evaluation_dir = output_dir / "evaluations"
    for path in [best_dir, checkpoint_dir, monitor_dir, evaluation_dir]:
        path.mkdir(parents=True, exist_ok=True)

    CommandEnv = load_env_class()
    train_env = Monitor(CommandEnv(), filename=str(monitor_dir / "train"))
    eval_env = Monitor(CommandEnv(), filename=str(monitor_dir / "eval"))
    train_env.reset(seed=SEED)
    eval_env.reset(seed=SEED + 10_000)

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="td3_command",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(evaluation_dir),
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ])

    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        seed=SEED,
        device="cpu",
        verbose=1,
    )
    print(f"Training TD3: steps={total_steps}, seed={SEED}, output={output_dir}")
    model.learn(total_timesteps=total_steps, callback=callbacks, log_interval=10)

    final_model = output_dir / "final_model.zip"
    model.save(final_model)
    train_env.close()
    eval_env.close()
    write_training_summary(monitor_dir / "train.monitor.csv", output_dir / "training_summary.csv")

    loaded = TD3.load(final_model, device="cpu")
    validation_env = CommandEnv()
    obs, info = validation_env.reset(seed=SEED)
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (8,) and np.isfinite(action).all()
    assert 0.0 <= info["command_vx"] <= 0.20 and -1.0 <= info["command_yaw"] <= 1.0
    validation_env.close()
    print(f"G2 {'SMOKE ' if args.smoke else ''}PASS: save/load/callback/evaluation pipeline complete")


if __name__ == "__main__":
    main()
