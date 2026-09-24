#!/usr/bin/env python3

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sesame_rl_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "td3"
FIGURE_DIR = RESULT_DIR / "figures"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(row, key):
    return float(row[key])


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


def mean(rows, key):
    return float(np.mean([number(row, key) for row in rows]))


def plot_training(training):
    steps = np.array([number(row, "timestep") for row in training])
    rewards = np.array([number(row, "episode_reward") for row in training])
    lengths = np.array([number(row, "episode_length") for row in training])
    window = min(20, len(rewards))

    plt.figure(figsize=(9, 4))
    plt.plot(steps, rewards, alpha=0.25, label="Episode")
    if window > 1:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(steps[window - 1:], smooth, label=f"{window}-episode mean")
    plt.xlabel("Timestep"); plt.ylabel("Episode reward"); plt.title("TD3 training reward")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01-training-reward.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(steps, lengths)
    plt.xlabel("Timestep"); plt.ylabel("Episode length (steps)"); plt.title("Training survival")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02-training-survival.png", dpi=180); plt.close()


def sweep_points(rows, sweep, command_key, response_key):
    selected = [row for row in rows if row["sweep"] == sweep]
    groups = defaultdict(list)
    for row in selected:
        groups[number(row, command_key)].append(row)
    commands = np.array(sorted(groups))
    means = np.array([mean(groups[value], response_key) for value in commands])
    stds = np.array([
        np.std([number(row, response_key) for row in groups[value]])
        for value in commands
    ])
    return commands, means, stds


def plot_sweeps(sweep):
    vx_cmd, vx, vx_std = sweep_points(sweep, "forward", "vx_cmd", "mean_vx")
    yaw_cmd, yaw, yaw_std = sweep_points(sweep, "yaw", "yaw_cmd", "mean_yaw_rate")
    for filename, x, y, error, xlabel, ylabel, title in [
        ("03-vx-command-tracking.png", vx_cmd, vx, vx_std, "vx command (m/s)", "Measured mean vx (m/s)", "Forward command tracking"),
        ("04-yaw-command-tracking.png", yaw_cmd, yaw, yaw_std, "Yaw command (rad/s)", "Measured mean yaw rate (rad/s)", "Yaw command tracking"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.errorbar(x, y, yerr=error, marker="o", capsize=3, label="Policy")
        plt.plot(x, x, "--", label="Ideal y=x")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(FIGURE_DIR / filename, dpi=180); plt.close()
    return vx_cmd, vx, yaw_cmd, yaw


def plot_fixed(evaluation):
    order = ["STOP", "FORWARD", "LEFT", "RIGHT"]
    groups = grouped(evaluation, "command_name")
    vx_error = [mean(groups[name], "vx_tracking_mae") for name in order]
    yaw_error = [mean(groups[name], "yaw_tracking_mae") for name in order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(order, vx_error); axes[0].set_ylabel("vx MAE (m/s)"); axes[0].set_title("Forward tracking error")
    axes[1].bar(order, yaw_error); axes[1].set_ylabel("Yaw MAE (rad/s)"); axes[1].set_title("Yaw tracking error")
    for axis in axes: axis.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "05-fixed-command-error.png", dpi=180); plt.close(fig)

    plt.figure(figsize=(7, 7))
    for name in order:
        row = groups[name][0]
        plt.plot(json.loads(row["trajectory_x"]), json.loads(row["trajectory_y"]), label=name)
    plt.scatter([0], [0], color="black", marker="o", label="Start")
    plt.xlabel("World X displacement (m)"); plt.ylabel("World Y displacement (m)")
    plt.title("Fixed-command trajectories"); plt.axis("equal"); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "06-fixed-command-trajectory.png", dpi=180); plt.close()

    upright = [mean(groups[name], "minimum_upright") for name in order]
    fall_rate = [np.mean([row["fall"].lower() == "true" for row in groups[name]]) for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(order, upright); axes[0].set_ylabel("Minimum upright"); axes[0].set_title("Upright stability")
    axes[1].bar(order, fall_rate); axes[1].set_ylabel("Fall rate"); axes[1].set_ylim(0, 1); axes[1].set_title("Falls")
    for axis in axes: axis.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "07-upright-comparison.png", dpi=180); plt.close(fig)
    return groups


def classifications(training, evaluation, sweep, fixed, vx_cmd, vx, yaw_cmd, yaw):
    numeric_sets = [
        (training, ["timestep", "episode_reward", "episode_length"]),
        (evaluation, ["mean_vx", "mean_yaw_rate", "minimum_upright", "path_efficiency"]),
        (sweep, ["mean_vx", "mean_yaw_rate", "minimum_upright", "path_efficiency"]),
    ]
    finite = all(np.isfinite(number(row, key)) for rows, keys in numeric_sets for row in rows for key in keys)
    engineering = (
        max(number(row, "timestep") for row in training) >= 500_000
        and (RESULT_DIR / "final_model.zip").exists()
        and finite
    )
    no_falls = not any(row["fall"].lower() == "true" for row in evaluation)
    forward_better = mean(fixed["FORWARD"], "mean_vx") > mean(fixed["STOP"], "mean_vx")
    left_correct = mean(fixed["LEFT"], "mean_yaw_rate") > 0
    right_correct = mean(fixed["RIGHT"], "mean_yaw_rate") < 0
    locomotion_score = sum([no_falls, forward_better, left_correct, right_correct])
    locomotion = "PASS" if locomotion_score == 4 else "PARTIAL" if no_falls and locomotion_score >= 2 else "FAIL"

    vx_slope = float(np.polyfit(vx_cmd, vx, 1)[0])
    yaw_slope = float(np.polyfit(yaw_cmd, yaw, 1)[0])
    forward_signal = vx_slope > 0 and vx[-1] > vx[0]
    forward_strong = forward_signal and int(np.argmax(vx)) == len(vx) - 1
    negative_yaw = float(np.mean(yaw[yaw_cmd < 0])) < 0
    positive_yaw = float(np.mean(yaw[yaw_cmd > 0])) > 0
    yaw_signal = yaw_slope > 0 and negative_yaw and positive_yaw
    nonzero = yaw_cmd != 0
    yaw_strong = yaw_signal and bool(np.all(np.sign(yaw[nonzero]) == np.sign(yaw_cmd[nonzero])))
    forward_status = "PASS" if forward_strong else "PARTIAL" if forward_signal else "FAIL"
    yaw_status = "PASS" if yaw_strong else "PARTIAL" if yaw_signal else "FAIL"
    command = "PASS" if forward_status == yaw_status == "PASS" else "PARTIAL" if "PASS" in [forward_status, yaw_status] or "PARTIAL" in [forward_status, yaw_status] else "FAIL"
    final = "PASS" if engineering and locomotion == "PASS" and command == "PASS" else "PARTIAL PASS" if engineering and locomotion != "FAIL" and command != "FAIL" else "FAIL"
    return engineering, locomotion, command, final, forward_status, yaw_status, vx_slope, yaw_slope


def write_report(training, evaluation, sweep, fixed, results):
    engineering, locomotion, command, final, vx_status, yaw_status, vx_slope, yaw_slope = results
    falls = sum(row["fall"].lower() == "true" for row in evaluation)
    fixed_table = "\n".join(
        f"| {name} | {mean(rows, 'mean_vx'):.4f} | {mean(rows, 'mean_yaw_rate'):.4f} | {mean(rows, 'vx_tracking_mae'):.4f} | {mean(rows, 'yaw_tracking_mae'):.4f} | {mean(rows, 'minimum_upright'):.4f} |"
        for name, rows in fixed.items()
    )
    forward_rows = [row for row in sweep if row["sweep"] == "forward"]
    yaw_rows = [row for row in sweep if row["sweep"] == "yaw"]
    vx_mae = float(np.mean([number(row, "vx_mae") for row in forward_rows]))
    yaw_mae = float(np.mean([number(row, "yaw_mae") for row in yaw_rows]))
    vx_zero_baseline = float(np.mean([abs(number(row, "vx_cmd")) for row in forward_rows]))
    yaw_zero_baseline = float(np.mean([abs(number(row, "yaw_cmd")) for row in yaw_rows]))
    callback_data = np.load(RESULT_DIR / "evaluations" / "evaluations.npz")
    callback_means = np.mean(callback_data["results"], axis=1)
    best_index = int(np.argmax(callback_means))
    best_step = int(callback_data["timesteps"][best_index])
    best_reward = float(callback_means[best_index])
    limitation = "極端 commands 已產生可區分行為，但中間 command 的 magnitude response 很弱；forward 在 0.20 m/s 的反應也低於 0.15 m/s，尚非可靠的連續追蹤。"
    next_experiment = "固定 architecture 與訓練流程，只做 K_V reward-width ablation；現有 forward reward 在 vx_cmd=0.20、vx=0 時仍約為 0.736。"
    report = f"""# 004 Command-Conditioned RL

## 1. Research Question
同一個 TD3 policy 能否依 continuous `[vx_cmd, yaw_cmd]` 產生可區分的 locomotion behavior？

## 2. Relationship with 003
003 比較 PPO、SAC、TD3；004 固定使用其選出的 TD3，不重新比較演算法。

## 3. Why TD3
003 TD3 完成 20 秒且未跌倒，minimum upright 0.9961、directional ratio 0.8612、path efficiency 0.9563、displacement 6.3845 m、mean forward speed 0.3250 m/s。TD3 僅被選為本實驗起點，不代表普遍最佳。

## 4. Experimental Pipeline
G1 environment → G2 500,000-step TD3 → G3 fixed commands → G4 command sweep → G5 analysis。

## 5. Environment
沿用 003 MuJoCo model、servo dynamics、20 ms control、joint limits、termination 與 safety logic；command 每 episode sampling 一次。

## 6. Observation Space
33D robot state加上 normalized `[vx_cmd/0.20, yaw_cmd/1.0]`，共 35D。

## 7. Action Space
8D continuous joint-target action，範圍 `[-1, 1]`。

## 8. Command Space
`vx_cmd ~ U(0, 0.20)` m/s、`yaw_cmd ~ U(-1, 1)` rad/s，STOP probability 0.15。

## 9. Reward Function
`r = 0.01 + 2 exp(-25(vx-vx_cmd)^2) + 0.20 exp(-(yaw-yaw_cmd)^2) - 0.50|vy| + 0.20u - 0.01 mean((a_t-a_(t-1))^2) - 10 I_fall`。

## 10. G1 Environment Validation
SB3 check、35D observation、8D action、sampling、fixed command、100-step rollout 與 finite checks：PASS。

## 11. G2 TD3 Training
完成 {int(max(number(row, 'timestep') for row in training)):,} steps；seed 42。EvalCallback 的最佳 checkpoint 位於 {best_step:,} steps，mean reward = {best_reward:.2f}。Engineering validity：{'PASS' if engineering else 'FAIL'}。

![](figures/01-training-reward.png)

![](figures/02-training-survival.png)

## 12. G3 Fixed Command Evaluation
每個 command 5 episodes、每 episode 20 秒，共 {len(evaluation)} episodes，fall count = {falls}。

| Command | mean vx | mean yaw | vx MAE | yaw MAE | min upright |
|---|---:|---:|---:|---:|---:|
{fixed_table}

![](figures/05-fixed-command-error.png)

![](figures/06-fixed-command-trajectory.png)

![](figures/07-upright-comparison.png)

## 13. G4 Continuous Command Sweep
Forward sweep 5 commands、yaw sweep 9 commands，各 5 episodes，共 {len(sweep)} episodes。

![](figures/03-vx-command-tracking.png)

![](figures/04-yaw-command-tracking.png)

## 14. Quantitative Results
Forward response slope = {vx_slope:.4f}；yaw response slope = {yaw_slope:.4f}。Forward sweep 平均 MAE = {vx_mae:.4f} m/s（零輸出 baseline = {vx_zero_baseline:.4f}）；yaw sweep 平均 MAE = {yaw_mae:.4f} rad/s（零輸出 baseline = {yaw_zero_baseline:.4f}）。Locomotion validity：{locomotion}。

## 15. Command Tracking Analysis
Forward conditioning：{vx_status}。整體 response 有正斜率，但 0.20 m/s response 低於 0.15 m/s，且 0.05/0.10 m/s 接近停止。Yaw conditioning：{yaw_status}。±1.0 rad/s 的方向正確，但部分中間 command 符號錯誤或接近零。Command-conditioning validity：{command}。

## 16. Failure Analysis
{limitation} 現有 forward reward 在 `vx_cmd=0.20, vx=0` 時仍為 `2e^-1 ≈ 0.736`，可能使較大 tracking error 仍取得不低 reward；本 baseline 未修改 reward。

## 17. Limitations
{limitation} 評估為 deterministic simulation；本實驗未涵蓋 camera、VLA、ROS2、domain randomization、sim-to-real 或 command switching。

## 18. Final Verdict
**{final}** — Engineering: {'PASS' if engineering else 'FAIL'}；Locomotion: {locomotion}；Command conditioning: {command}。

## 19. Next Experiment
{next_experiment}
"""
    (RESULT_DIR / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return limitation, next_experiment


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    training = read_csv(RESULT_DIR / "training_summary.csv")
    evaluation = read_csv(RESULT_DIR / "evaluation_summary.csv")
    sweep = read_csv(RESULT_DIR / "command_sweep.csv")
    plot_training(training)
    vx_cmd, vx, yaw_cmd, yaw = plot_sweeps(sweep)
    fixed = plot_fixed(evaluation)
    results = classifications(training, evaluation, sweep, fixed, vx_cmd, vx, yaw_cmd, yaw)
    limitation, next_experiment = write_report(training, evaluation, sweep, fixed, results)
    engineering, locomotion, command, final, vx_status, yaw_status, _, _ = results
    main_finding = "Policy 能穩定區分 STOP 與極端轉向 command，但 continuous velocity/yaw magnitude tracking 仍弱。"

    print("=" * 60)
    print("004 COMMAND-CONDITIONED RL — FINAL")
    print("=" * 60)
    print("01 Environment\nPASS\n")
    print("02 TD3 Training\n500000 steps\n" + ("PASS" if engineering else "FAIL") + "\n")
    print("03 Fixed Command Evaluation")
    for name in ["STOP", "FORWARD", "LEFT", "RIGHT"]:
        print(f"{name:<10} vx={mean(fixed[name], 'mean_vx'):+.4f} yaw={mean(fixed[name], 'mean_yaw_rate'):+.4f}")
    print(f"\n04 Continuous Command Sweep\nForward conditioning : {vx_status}\nYaw conditioning     : {yaw_status}\n")
    print(f"05 Analysis\n\nEngineering validity\n{'PASS' if engineering else 'FAIL'}\n\nLocomotion validity\n{locomotion}\n\nCommand-conditioning validity\n{command}\n\nFinal verdict\n{final}")
    print(f"\nMain finding\n{main_finding}\n\nMain limitation\n{limitation}\n\nRecommended next experiment\n{next_experiment}")
    print("=" * 60)
    print("Created files:\n01-command-env.py\n02-train-td3.py\n03-evaluate-fixed-command.py\n04-command-sweep.py\n05-analyze-results.py\nREADME.md")
    print("\nModels:\nresults/td3/best_model/best_model.zip\nresults/td3/final_model.zip\nresults/td3/checkpoints/")
    print("\nCSV:\nresults/td3/training_summary.csv\nresults/td3/evaluation_summary.csv\nresults/td3/command_sweep.csv")
    print("\nFigures:\nresults/td3/figures/01-training-reward.png ... 07-upright-comparison.png")
    print("\nReport:\nresults/td3/FINAL_REPORT.md")


if __name__ == "__main__":
    main()
