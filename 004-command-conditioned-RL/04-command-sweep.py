#!/usr/bin/env python3

import csv
import importlib.util
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "td3"
SWEEP_DIR = RESULT_DIR / "command_sweep"
EPISODES = 5
COMMANDS = (
    [("forward", vx, 0.0) for vx in [0.00, 0.05, 0.10, 0.15, 0.20]]
    + [("yaw", 0.10, yaw) for yaw in np.linspace(-1.0, 1.0, 9)]
)


def load_env_class():
    spec = importlib.util.spec_from_file_location("command_env", ROOT / "01-command-env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CommandEnv


def model_path():
    best = RESULT_DIR / "best_model" / "best_model.zip"
    final = RESULT_DIR / "final_model.zip"
    if best.exists():
        return best
    if final.exists():
        print("Best model missing; using final_model.zip")
        return final
    raise FileNotFoundError("No trained TD3 model found")


def rollout(env, model, sweep, vx_cmd, yaw_cmd, episode):
    env.set_command(vx_cmd, yaw_cmd)
    obs, _ = env.reset(seed=30_000 + episode)
    start = env.data.xpos[env.base_body_id].copy()
    previous = start.copy()
    vx_values, yaw_values, upright_values = [], [], []
    path_length = 0.0
    terminated = truncated = False
    for _ in range(1_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        position = env.data.xpos[env.base_body_id].copy()
        path_length += float(np.linalg.norm(position[:2] - previous[:2]))
        previous = position
        vx_values.append(info["forward_velocity"])
        yaw_values.append(info["yaw_rate"])
        upright_values.append(info["upright"])
        if terminated or truncated:
            break
    relative = previous[:2] - start[:2]
    displacement = float(np.linalg.norm(relative))
    return {
        "sweep": sweep,
        "vx_cmd": vx_cmd,
        "yaw_cmd": yaw_cmd,
        "episode": episode,
        "mean_vx": float(np.mean(vx_values)),
        "mean_yaw_rate": float(np.mean(yaw_values)),
        "vx_mae": float(np.mean(np.abs(np.asarray(vx_values) - vx_cmd))),
        "yaw_mae": float(np.mean(np.abs(np.asarray(yaw_values) - yaw_cmd))),
        "minimum_upright": float(np.min(upright_values)),
        "fall": bool(terminated),
        "displacement": displacement,
        "lateral_drift": abs(float(relative[1])),
        "path_efficiency": displacement / path_length if path_length else 0.0,
    }


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    CommandEnv = load_env_class()
    env = CommandEnv()
    model = TD3.load(model_path(), device="cpu")
    rows = [
        rollout(env, model, sweep, vx, yaw, episode)
        for sweep, vx, yaw in COMMANDS
        for episode in range(1, EPISODES + 1)
    ]
    env.close()
    output = RESULT_DIR / "command_sweep.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"G4 complete: {len(rows)} episodes -> {output}")


if __name__ == "__main__":
    main()
