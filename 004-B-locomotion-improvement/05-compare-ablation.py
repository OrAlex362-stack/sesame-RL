#!/usr/bin/env python3

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sesame_rl_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
BASELINE_MODEL = ROOT.parent / "004-command-conditioned-RL" / "results" / "td3" / "best_model" / "best_model.zip"
K_VALUES = [25, 50, 75, 100]


def read_csv(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def value(row, key):
    return float(row[key])


def group_mean(rows, group_key, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[value(row, group_key)].append(value(row, metric))
    keys = np.array(sorted(groups))
    means = np.array([np.mean(groups[key]) for key in keys])
    stds = np.array([np.std(groups[key]) for key in keys])
    return keys, means, stds


def metrics(k_v):
    result = RESULTS / f"kv{k_v}"
    sweep = read_csv(result / "command_sweep.csv")
    fixed = read_csv(result / "evaluation_summary.csv")
    forward = [row for row in sweep if row["sweep"] == "forward"]
    yaw = [row for row in sweep if row["sweep"] == "yaw"]
    vx_cmd, measured_vx, vx_std = group_mean(forward, "vx_cmd", "mean_vx")
    yaw_cmd, measured_yaw, yaw_std = group_mean(yaw, "yaw_cmd", "mean_yaw_rate")
    slope, intercept = np.polyfit(vx_cmd, measured_vx, 1)
    yaw_slope = float(np.polyfit(yaw_cmd, measured_yaw, 1)[0])
    separations = np.diff(measured_vx)
    fixed_groups = defaultdict(list)
    for row in fixed:
        fixed_groups[row["command_name"]].append(row)
    fixed_mean = lambda name, key: float(np.mean([value(row, key) for row in fixed_groups[name]]))
    engineering = BASELINE_MODEL.exists() if k_v == 25 else (
        (result / "best_model" / "best_model.zip").exists()
        and (result / "final_model.zip").exists()
        and len(read_csv(result / "training_summary.csv")) == 500
    )
    numeric = [
        *[value(row, key) for row in sweep for key in ["mean_vx", "mean_yaw_rate", "vx_mae", "yaw_mae"]],
        *[value(row, key) for row in fixed for key in ["minimum_upright", "path_efficiency", "lateral_drift"]],
    ]
    engineering = bool(engineering and np.isfinite(numeric).all())
    fall_rate = float(np.mean([row["fall"].lower() == "true" for row in fixed]))
    left_sign = fixed_mean("LEFT", "mean_yaw_rate") > 0
    right_sign = fixed_mean("RIGHT", "mean_yaw_rate") < 0
    locomotion = bool(
        engineering
        and fall_rate == 0.0
        and fixed_mean("FORWARD", "mean_vx") > fixed_mean("STOP", "mean_vx")
        and left_sign
        and right_sign
    )
    return {
        "K_V": k_v,
        "forward_mae": float(np.mean([value(row, "vx_mae") for row in forward])),
        "forward_response_slope": float(slope),
        "forward_intercept": float(intercept),
        "monotonic_violations": int(np.sum(separations < 0)),
        "mean_command_separation": float(np.mean(np.abs(separations))),
        "fall_rate": fall_rate,
        "minimum_upright": float(np.min([value(row, "minimum_upright") for row in fixed])),
        "path_efficiency": float(np.mean([value(row, "path_efficiency") for row in fixed])),
        "lateral_drift": float(np.mean([value(row, "lateral_drift") for row in fixed])),
        "mean_episode_length": float(np.mean([value(row, "episode_length") for row in fixed])),
        "yaw_mae": float(np.mean([value(row, "yaw_mae") for row in yaw])),
        "yaw_response_slope": yaw_slope,
        "left_sign_correct": left_sign,
        "right_sign_correct": right_sign,
        "engineering_valid": engineering,
        "locomotion_valid": locomotion,
        "vx_cmd": vx_cmd,
        "measured_vx": measured_vx,
        "vx_std": vx_std,
        "yaw_cmd": yaw_cmd,
        "measured_yaw": measured_yaw,
        "yaw_std": yaw_std,
        "separations": separations,
        "fixed": fixed_groups,
    }


def classify(all_metrics):
    baseline = all_metrics[0]
    for item in all_metrics:
        if item["K_V"] == 25:
            item["classification"] = "BASELINE"
            item["primary_improvements"] = 0
            continue
        checks = [
            item["forward_mae"] < baseline["forward_mae"],
            abs(item["forward_response_slope"] - 1) < abs(baseline["forward_response_slope"] - 1),
            item["monotonic_violations"] < baseline["monotonic_violations"],
            item["mean_command_separation"] > baseline["mean_command_separation"],
        ]
        guardrail = item["locomotion_valid"] and item["fall_rate"] <= baseline["fall_rate"]
        yaw_guardrail = item["left_sign_correct"] and item["right_sign_correct"]
        item["primary_improvements"] = sum(checks)
        item["classification"] = (
            "IMPROVED" if all(checks) and guardrail and yaw_guardrail
            else "MIXED" if any(checks) and guardrail and yaw_guardrail
            else "REGRESSED"
        )

    improved = [item for item in all_metrics if item["classification"] == "IMPROVED"]
    mixed = [item for item in all_metrics if item["classification"] == "MIXED"]
    if improved:
        candidate = min(improved, key=lambda item: (item["forward_mae"], abs(item["forward_response_slope"] - 1)))
        verdict = "SUPPORTED"
    else:
        partial = [
            item for item in mixed
            if item["forward_mae"] < baseline["forward_mae"]
            and abs(item["forward_response_slope"] - 1) < abs(baseline["forward_response_slope"] - 1)
        ]
        candidate = min(partial, key=lambda item: (-item["primary_improvements"], item["forward_mae"])) if partial else None
        verdict = "PARTIALLY SUPPORTED" if partial else "NOT SUPPORTED"
    return candidate, verdict


def make_figures(items):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for item in items:
        plt.errorbar(item["vx_cmd"], item["measured_vx"], yerr=item["vx_std"], marker="o", capsize=3, label=f"K{item['K_V']}")
    plt.plot([0, 0.20], [0, 0.20], "k--", label="Ideal y=x")
    plt.xlabel("vx command (m/s)"); plt.ylabel("Measured mean vx (m/s)")
    plt.title("Forward command response across reward widths")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "02-vx-command-response-all-kv.png", dpi=180); plt.close()

    ks = [item["K_V"] for item in items]
    for filename, metric, ylabel, reference in [
        ("03-vx-mae-vs-kv.png", "forward_mae", "Forward MAE (m/s)", None),
        ("04-response-slope-vs-kv.png", "forward_response_slope", "Forward response slope", 1.0),
    ]:
        plt.figure(figsize=(7, 4))
        plt.plot(ks, [item[metric] for item in items], marker="o")
        if reference is not None: plt.axhline(reference, linestyle="--", color="black", label="Ideal")
        plt.xlabel("K_V"); plt.ylabel(ylabel); plt.title(ylabel + " vs K_V")
        plt.grid(alpha=0.3)
        if reference is not None: plt.legend()
        plt.tight_layout(); plt.savefig(FIGURES / filename, dpi=180); plt.close()

    labels = ["0→0.05", "0.05→0.10", "0.10→0.15", "0.15→0.20"]
    x = np.arange(len(labels)); width = 0.19
    plt.figure(figsize=(9, 5))
    for index, item in enumerate(items):
        plt.bar(x + (index - 1.5) * width, item["separations"], width, label=f"K{item['K_V']}")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels); plt.ylabel("Δ measured vx (m/s)")
    plt.title("Adjacent forward-command separation"); plt.grid(axis="y", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "05-forward-command-separation.png", dpi=180); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric, title in zip(axes.flat, ["fall_rate", "minimum_upright", "path_efficiency", "lateral_drift"], ["Fall rate", "Minimum upright", "Path efficiency", "Lateral drift (m)"]):
        axis.plot(ks, [item[metric] for item in items], marker="o"); axis.set_title(title); axis.set_xlabel("K_V"); axis.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "06-stability-vs-kv.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, [item["yaw_mae"] for item in items], marker="o"); axes[0].set_ylabel("Yaw MAE (rad/s)")
    axes[1].plot(ks, [item["yaw_response_slope"] for item in items], marker="o"); axes[1].set_ylabel("Yaw response slope")
    for axis in axes: axis.set_xlabel("K_V"); axis.grid(alpha=0.3)
    fig.suptitle("Yaw guardrail"); fig.tight_layout(); fig.savefig(FIGURES / "07-yaw-guardrail-vs-kv.png", dpi=180); plt.close(fig)


def write_summary(items):
    fields = [
        "K_V", "forward_mae", "forward_response_slope", "forward_intercept",
        "monotonic_violations", "mean_command_separation", "fall_rate", "minimum_upright",
        "path_efficiency", "lateral_drift", "mean_episode_length", "yaw_mae",
        "yaw_response_slope", "left_sign_correct", "right_sign_correct",
        "engineering_valid", "locomotion_valid", "classification",
    ]
    with (RESULTS / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({key: item[key] for key in fields})


def write_report(items, candidate, verdict):
    baseline = items[0]
    table = "\n".join(
        f"| {item['K_V']} | {item['forward_mae']:.4f} | {item['forward_response_slope']:.4f} | {item['monotonic_violations']} | {item['mean_command_separation']:.4f} | {item['fall_rate']:.2f} | {item['minimum_upright']:.4f} | {item['classification']} |"
        for item in items
    )
    candidate_text = f"K_V={candidate['K_V']}" if candidate else "無符合全部條件的 candidate"
    if verdict == "SUPPORTED":
        conclusion = f"Reward-width hypothesis supported；{candidate_text} 同時改善主要 forward metrics 且通過 stability/yaw guardrails。"
        recommendation = f"先將 K_V={candidate['K_V']} 視為 005 locomotion backend candidate，進入多 seed robustness validation 後再定案。"
    elif verdict == "PARTIALLY SUPPORTED":
        conclusion = f"Reward-width hypothesis partially supported；{candidate_text} 改善部分 tracking metrics，但仍有 monotonicity、separation 或 guardrail trade-off。"
        recommendation = "005 暫不替換 K25 baseline；先對候選值完成多 seed robustness validation。"
    else:
        conclusion = "Reward-width hypothesis not supported；單獨縮窄 forward reward 不足以解決 continuous tracking。"
        recommendation = "005 保留 K25 baseline；下一個 hypothesis 應轉向 command sampling、reward structure 或 policy representation。"
    report = f"""# 004-B Locomotion Improvement
## K_V Reward-Width Ablation

## 1. Motivation
004 的 continuous forward tracking 僅達 PARTIAL，低速 commands 接近停止且 0.20 m/s response 低於 0.15 m/s。

## 2. Baseline from 004
TD3、500,000 steps、seed 42；K_V=25 的 forward MAE={baseline['forward_mae']:.4f} m/s、response slope={baseline['forward_response_slope']:.4f}。

## 3. Hypothesis
K_V=25 的 reward width 太寬；在 0.20 m/s error 時仍提供約 0.736 reward，可能不要求 policy 精確追蹤速度。

## 4. Controlled Variable
唯一變數為 K_V：25、50、75、100。TD3、seed、500k budget、35D observation、8D action、commands、K_YAW=1、其他 reward weights、physics 與 evaluation 全部固定。

## 5. Reward Width Analysis
![](figures/01-reward-width-comparison.png)

最大 error 0.20 m/s 時，K25/K50/K75/K100 的 forward rewards 約為 0.736/0.271/0.100/0.037。

## 6. Experimental Design
K25 直接引用 004；K50、K75、K100 各訓練一次 500,000 steps。所有比較使用 best_model 與相同 fixed/sweep protocols。

## 7. Training
三個新 variants 均使用相同 TD3 configuration。Training reward 因 reward landscape 不同，不用於 winner selection。

## 8. Fixed Command Evaluation
每個 variant 對 STOP、FORWARD、LEFT、RIGHT 各執行 5 個 20 秒 episodes；stability 與 turning signs 作為 guardrails。

## 9. Continuous Forward Sweep
![](figures/02-vx-command-response-all-kv.png)

## 10. Quantitative Comparison
| K_V | Forward MAE | Slope | Monotonic violations | Mean separation | Fall rate | Min upright | Classification |
|---:|---:|---:|---:|---:|---:|---:|---|
{table}

![](figures/03-vx-mae-vs-kv.png)

![](figures/04-response-slope-vs-kv.png)

![](figures/05-forward-command-separation.png)

## 11. Stability Guardrails
![](figures/06-stability-vs-kv.png)

## 12. Yaw Guardrail
Yaw 未被最佳化；僅檢查 yaw MAE/slope 與 LEFT/RIGHT sign，避免 narrowing forward reward 造成 turning collapse。

![](figures/07-yaw-guardrail-vs-kv.png)

## 13. Ablation Result
**{conclusion}** Best forward tracking candidate：{candidate_text}。

## 14. Failure / Trade-off Analysis
Winner 不由 training reward 決定；必須同時改善 MAE、slope、monotonicity、separation，且無 locomotion/yaw collapse。分類依實際 physical metrics，不為取得正結果而修改 threshold 或重訓。

## 15. Limitations
第一輪為 single-seed screening；deterministic simulation 不代表多 seed robustness 或 sim-to-real performance。

## 16. Conclusion
{conclusion}

## 17. Recommendation for 005
{recommendation}
"""
    (RESULTS / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return candidate_text, conclusion, recommendation


def main():
    items = [metrics(k_v) for k_v in K_VALUES]
    candidate, verdict = classify(items)
    make_figures(items)
    write_summary(items)
    candidate_text, conclusion, recommendation = write_report(items, candidate, verdict)

    print("=" * 60); print("004-B K_V REWARD-WIDTH ABLATION"); print("=" * 60)
    for item in items:
        label = "Baseline\n" if item["K_V"] == 25 else ""
        print(f"\n{label}K_V = {item['K_V']}\nForward MAE           : {item['forward_mae']:.4f}\nResponse slope        : {item['forward_response_slope']:.4f}\nMonotonic violations  : {item['monotonic_violations']}\nFall rate             : {item['fall_rate']:.2f}\nMin upright           : {item['minimum_upright']:.4f}\nYaw guardrail         : {'PASS' if item['left_sign_correct'] and item['right_sign_correct'] else 'FAIL'}\nClassification        : {item['classification']}")
        if item["K_V"] == 25: print("\n" + "-" * 60)
    print("\n" + "-" * 60)
    print(f"\nBest forward tracking candidate:\n{candidate_text}\n\nReward-width hypothesis:\n{verdict}\n\nMain improvement:\n{conclusion}\n\nMain trade-off:\nSingle-seed result; yaw and stability remain guardrails, not optimized targets.\n\nRecommended policy for 005:\n{recommendation}\n\nNeed additional tuning?\nYES")
    print("\n" + "=" * 60)
    print("Created files:\n01-reward-width-analysis.py\n02-train-ablation.py\n03-evaluate-ablation.py\n04-command-sweep.py\n05-compare-ablation.py\nREADME.md")
    print("\nModels:\nresults/kv50|kv75|kv100/best_model/ and final_model.zip")
    print("\nCSV:\nresults/reward_width_table.csv\nresults/kv*/training_summary.csv\nresults/kv*/evaluation_summary.csv\nresults/kv*/command_sweep.csv\nresults/ablation_summary.csv")
    print("\nFigures:\nresults/figures/01-reward-width-comparison.png ... 07-yaw-guardrail-vs-kv.png")
    print("\nReport:\nresults/FINAL_REPORT.md")


if __name__ == "__main__":
    main()
