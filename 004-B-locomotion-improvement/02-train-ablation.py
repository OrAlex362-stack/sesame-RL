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
BASE_ENV = ROOT.parent / "004-command-conditioned-RL" / "01-command-env.py"
RESULTS = ROOT / "results"
MAIN_K_VALUES = [50, 75, 100]
TOTAL_TIMESTEPS = 500_000
SEED = 42


def load_env_class(k_v):
    spec = importlib.util.spec_from_file_location(f"command_env_kv{k_v}", BASE_ENV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.K_V = float(k_v)
    assert module.K_YAW == 1.0 and module.STOP_PROBABILITY == 0.15
    return module.CommandEnv


def write_training_summary(monitor_file, evaluations_file, output_file):
    with monitor_file.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    evaluations = np.load(evaluations_file)
    means = np.mean(evaluations["results"], axis=1)
    best_index = int(np.argmax(means))
    best_step = int(evaluations["timesteps"][best_index])
    best_reward = float(means[best_index])
    timestep = 0
    with output_file.open("w", newline="", encoding="utf-8") as target:
        fields = [
            "episode", "timestep", "episode_reward", "episode_length", "elapsed_seconds",
            "terminated", "fall", "best_evaluation_reward", "best_checkpoint_timestep",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for episode, row in enumerate(rows, 1):
            timestep += int(row["l"])
            terminated = row.get("terminated", "False") == "True"
            writer.writerow({
                "episode": episode,
                "timestep": timestep,
                "episode_reward": row["r"],
                "episode_length": row["l"],
                "elapsed_seconds": row["t"],
                "terminated": terminated,
                "fall": terminated,
                "best_evaluation_reward": best_reward,
                "best_checkpoint_timestep": best_step,
            })
    return best_step, best_reward


def train(k_v, seed, smoke=False, robustness=False):
    if k_v == 25 and not robustness:
        raise ValueError("K_V=25 main result must reuse 004; retraining is forbidden")
    if smoke:
        output = RESULTS / "smoke" / f"kv{k_v}"
    elif robustness:
        output = RESULTS / "robustness" / f"kv{k_v}" / f"seed{seed}"
    else:
        output = RESULTS / f"kv{k_v}"
    if (output / "final_model.zip").exists():
        raise FileExistsError(f"Completed model already exists: {output / 'final_model.zip'}")

    total_steps = 5_000 if smoke else TOTAL_TIMESTEPS
    eval_freq = 1_000 if smoke else 10_000
    checkpoint_freq = 2_500 if smoke else 50_000
    best_dir = output / "best_model"
    checkpoint_dir = output / "checkpoints"
    monitor_dir = output / "monitor"
    evaluation_dir = output / "evaluations"
    for path in [best_dir, checkpoint_dir, monitor_dir, evaluation_dir]:
        path.mkdir(parents=True, exist_ok=True)

    CommandEnv = load_env_class(k_v)
    train_env = Monitor(
        CommandEnv(), filename=str(monitor_dir / "train"), info_keywords=("terminated",)
    )
    eval_env = Monitor(
        CommandEnv(), filename=str(monitor_dir / "eval"), info_keywords=("terminated",)
    )
    train_env.reset(seed=seed)
    eval_env.reset(seed=seed + 10_000)
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=checkpoint_freq, save_path=str(checkpoint_dir), name_prefix=f"td3_kv{k_v}"
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
        seed=seed,
        device="cpu",
        verbose=1,
    )
    print(f"Training K_V={k_v}: steps={total_steps}, seed={seed}, output={output}")
    model.learn(total_timesteps=total_steps, callback=callbacks, log_interval=10)
    final_model = output / "final_model.zip"
    model.save(final_model)
    train_env.close(); eval_env.close()
    best_step, best_reward = write_training_summary(
        monitor_dir / "train.monitor.csv",
        evaluation_dir / "evaluations.npz",
        output / "training_summary.csv",
    )

    loaded = TD3.load(final_model, device="cpu")
    validation_env = CommandEnv()
    obs, info = validation_env.reset(seed=seed)
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (8,) and np.isfinite(action).all()
    assert 0.0 <= info["command_vx"] <= 0.20 and -1.0 <= info["command_yaw"] <= 1.0
    validation_env.close()
    print(f"K_V={k_v} PASS: best_step={best_step}, best_eval_reward={best_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", type=int, choices=[25, 50, 75, 100])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    args = parser.parse_args()
    values = [args.kv] if args.kv is not None else MAIN_K_VALUES
    for k_v in values:
        train(k_v, args.seed, smoke=args.smoke, robustness=args.robustness)


if __name__ == "__main__":
    main()
