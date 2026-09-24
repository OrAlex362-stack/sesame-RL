#!/usr/bin/env python3

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

ROOT = Path(__file__).resolve().parent
BASE_ENV = ROOT.parent / "004-command-conditioned-RL" / "01-command-env.py"
RESULTS = ROOT / "results"
K_VALUES = [50, 75, 100]
EPISODES = 5
COMMANDS = (
    [("forward", vx, 0.0) for vx in [0.00, 0.05, 0.10, 0.15, 0.20]]
    + [("yaw", 0.10, yaw) for yaw in np.linspace(-1.0, 1.0, 9)]
)


def load_env_class(k_v):
    spec = importlib.util.spec_from_file_location(f"command_env_kv{k_v}", BASE_ENV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.K_V = float(k_v)
    return module.CommandEnv


def model_path(k_v):
    result = RESULTS / f"kv{k_v}"
    best = result / "best_model" / "best_model.zip"
    final = result / "final_model.zip"
    if best.exists():
        return best
    if final.exists():
        print(f"K_V={k_v}: best model missing; using final model")
        return final
    raise FileNotFoundError(f"No trained K_V={k_v} model")


def rollout(env, model, k_v, sweep, vx_cmd, yaw_cmd, episode):
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
        "K_V": k_v,
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
        "episode_length": len(vx_values),
        "displacement": displacement,
        "lateral_drift": abs(float(relative[1])),
        "path_efficiency": displacement / path_length if path_length else 0.0,
    }


def evaluate(k_v):
    CommandEnv = load_env_class(k_v)
    env = CommandEnv()
    model = TD3.load(model_path(k_v), device="cpu")
    rows = [
        rollout(env, model, k_v, sweep, vx, yaw, episode)
        for sweep, vx, yaw in COMMANDS
        for episode in range(1, EPISODES + 1)
    ]
    env.close()
    output = RESULTS / f"kv{k_v}" / "command_sweep.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"K_V={k_v}: command sweep {len(rows)} episodes -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv", type=int, choices=K_VALUES)
    args = parser.parse_args()
    for k_v in [args.kv] if args.kv else K_VALUES:
        evaluate(k_v)


if __name__ == "__main__":
    main()
