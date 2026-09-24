#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "td3"
EPISODES = 5
COMMANDS = {
    "STOP": (0.00, 0.00),
    "FORWARD": (0.20, 0.00),
    "LEFT": (0.10, 1.00),
    "RIGHT": (0.10, -1.00),
}


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


def rollout(env, model, name, command, episode):
    env.set_command(*command)
    obs, _ = env.reset(seed=20_000 + episode)
    start = env.data.xpos[env.base_body_id].copy()
    previous = start.copy()
    xs, ys = [0.0], [0.0]
    vx_values, vy_values, yaw_values, upright_values = [], [], [], []
    reward_total = path_length = 0.0
    terminated = truncated = False

    for _ in range(env.spec.max_episode_steps if env.spec else 1_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        position = env.data.xpos[env.base_body_id].copy()
        path_length += float(np.linalg.norm(position[:2] - previous[:2]))
        previous = position
        xs.append(float(position[0] - start[0]))
        ys.append(float(position[1] - start[1]))
        vx_values.append(info["forward_velocity"])
        vy_values.append(info["lateral_velocity"])
        yaw_values.append(info["yaw_rate"])
        upright_values.append(info["upright"])
        reward_total += reward
        if terminated or truncated:
            break

    final_x, final_y = xs[-1], ys[-1]
    rotation = env._get_base_rotation()
    final_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    displacement = float(np.hypot(final_x, final_y))
    steps = len(vx_values)
    return {
        "command_name": name,
        "episode": episode,
        "vx_cmd": command[0],
        "yaw_cmd": command[1],
        "episode_reward": reward_total,
        "episode_length": steps,
        "mean_vx": float(np.mean(vx_values)),
        "mean_vy": float(np.mean(vy_values)),
        "mean_yaw_rate": float(np.mean(yaw_values)),
        "vx_tracking_mae": float(np.mean(np.abs(np.asarray(vx_values) - command[0]))),
        "yaw_tracking_mae": float(np.mean(np.abs(np.asarray(yaw_values) - command[1]))),
        "minimum_upright": float(np.min(upright_values)),
        "mean_upright": float(np.mean(upright_values)),
        "fall": bool(terminated),
        "termination_reason": "fall" if terminated else "time_limit" if truncated else "max_steps",
        "final_x": final_x,
        "final_y": final_y,
        "final_yaw": final_yaw,
        "displacement": displacement,
        "path_length": path_length,
        "path_efficiency": displacement / path_length if path_length else 0.0,
        "lateral_drift": abs(final_y),
        "trajectory_x": json.dumps(xs),
        "trajectory_y": json.dumps(ys),
    }


def main():
    CommandEnv = load_env_class()
    env = CommandEnv()
    model = TD3.load(model_path(), device="cpu")
    rows = [
        rollout(env, model, name, command, episode)
        for name, command in COMMANDS.items()
        for episode in range(1, EPISODES + 1)
    ]
    env.close()
    output = RESULT_DIR / "evaluation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"G3 complete: {len(rows)} episodes -> {output}")


if __name__ == "__main__":
    main()
