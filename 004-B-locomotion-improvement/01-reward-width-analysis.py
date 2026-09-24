#!/usr/bin/env python3

import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sesame_rl_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BASELINE = PROJECT_ROOT / "004-command-conditioned-RL" / "results" / "td3"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
K_VALUES = [25, 50, 75, 100]
ERRORS = [0.00, 0.05, 0.10, 0.15, 0.20]


def reward(k_v, error):
    return 2.0 * np.exp(-k_v * np.asarray(error) ** 2)


def baseline_metrics():
    with (BASELINE / "command_sweep.csv").open(encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["sweep"] == "forward"]
    groups = {}
    for row in rows:
        groups.setdefault(float(row["vx_cmd"]), []).append(float(row["mean_vx"]))
    commands = np.array(sorted(groups))
    responses = np.array([np.mean(groups[value]) for value in commands])
    return {
        "forward_mae": float(np.mean([float(row["vx_mae"]) for row in rows])),
        "forward_response_slope": float(np.polyfit(commands, responses, 1)[0]),
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    kv25 = RESULTS / "kv25"
    kv25.mkdir(parents=True, exist_ok=True)

    dense_errors = np.linspace(-0.20, 0.20, 401)
    plt.figure(figsize=(8, 5))
    for k_v in K_VALUES:
        plt.plot(dense_errors, reward(k_v, dense_errors), label=f"K_V={k_v}")
    plt.xlabel("Forward tracking error (m/s)")
    plt.ylabel("Forward tracking reward")
    plt.title("Forward reward-width comparison")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "01-reward-width-comparison.png", dpi=180)
    plt.close()

    table = RESULTS / "reward_width_table.csv"
    with table.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tracking_error"] + [f"K{k_v}_reward" for k_v in K_VALUES])
        for error in ERRORS:
            writer.writerow([error] + [float(reward(k_v, error)) for k_v in K_VALUES])

    for name in ["training_summary.csv", "evaluation_summary.csv", "command_sweep.csv"]:
        shutil.copy2(BASELINE / name, kv25 / name)
    metrics = baseline_metrics()
    reference = {
        "K_V": 25,
        "algorithm": "TD3",
        "total_timesteps": 500_000,
        "seed": 42,
        "model": str(BASELINE / "best_model" / "best_model.zip"),
        **metrics,
    }
    (kv25 / "baseline_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    print(f"G1 complete: reward-width figure/table and K25 baseline reference -> {RESULTS}")


if __name__ == "__main__":
    main()
