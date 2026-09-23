#!/usr/bin/env python3

"""
04_compare_models.py
====================

Generic PPO policy comparison tool.

Only change MODEL_A / MODEL_B and LABEL_A / LABEL_B.

Outputs:
- 04_trajectory.png
- 04_forward_velocity.png
- 04_lateral_velocity.png
- 04_yaw_rate.png
- 04_metrics.png
- 04_summary.csv

Important:
Raw reward is NOT compared because different reward functions
may use different scales.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from sesame_rl_env import (
    SesameRLEnv,
    CONTROL_DT,
    EPISODE_STEPS,
)


# ============================================================
# 1. ONLY CHANGE THESE
# ============================================================

MODEL_A = (
    "/home/kit/sesame-RL/"
    "003_rl_results/01_ppo_baseline_001/"
    "best_model/best_model.zip"
)

MODEL_B = (
    "/home/kit/sesame-RL/"
    "003_rl_results/02_reward_v2_yaw/"
    "best_model/best_model.zip"
)

LABEL_A = "Reward V1"

LABEL_B = "Reward V2"


# ============================================================
# 2. CONFIG
# ============================================================

SEED = 2000

OUTPUT_DIR = Path(
    "/home/kit/sesame-RL/"
    "003_rl_results/model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. MODEL PATH HELPER
# ============================================================

def resolve_model_path(path):

    path = Path(path)

    if path.exists():
        return path

    # Allow path without .zip
    zip_path = Path(
        str(path) + ".zip"
    )

    if zip_path.exists():
        return zip_path

    raise FileNotFoundError(
        f"Model not found:\n{path}"
    )


# ============================================================
# 4. ROLLOUT
# ============================================================

def rollout(
    model_path,
    label,
):

    model_path = resolve_model_path(
        model_path
    )

    print()
    print(
        f"Loading {label}"
    )

    print(
        model_path
    )

    model = PPO.load(
        str(model_path)
    )

    # --------------------------------------------------------
    # Same environment for every model.
    #
    # Reward weight does not affect dynamics or observations,
    # so comparison focuses on physical behaviour.
    # --------------------------------------------------------

    env = SesameRLEnv()

    obs, _ = env.reset(
        seed=SEED
    )

    times = []

    x_values = []
    y_values = []

    forward_values = []
    lateral_values = []
    yaw_values = []

    upright_values = []

    start_position = (
        env.data.xpos[
            env.base_body_id
        ].copy()
    )

    previous_position = (
        start_position.copy()
    )

    path_length = 0.0

    terminated = False
    truncated = False


    # ========================================================
    # POLICY ROLLOUT
    # ========================================================

    for step in range(
        EPISODE_STEPS
    ):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        position = (
            env.data.xpos[
                env.base_body_id
            ].copy()
        )

        # ----------------------------------------------------
        # Path length
        # ----------------------------------------------------

        path_length += float(
            np.linalg.norm(
                position[:2]
                - previous_position[:2]
            )
        )

        previous_position = (
            position.copy()
        )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        times.append(
            step
            * CONTROL_DT
        )

        x_values.append(
            position[0]
            - start_position[0]
        )

        y_values.append(
            position[1]
            - start_position[1]
        )

        forward_values.append(
            float(
                info[
                    "forward_velocity"
                ]
            )
        )

        lateral_values.append(
            float(
                info[
                    "lateral_velocity"
                ]
            )
        )

        yaw_values.append(
            float(
                np.rad2deg(
                    info[
                        "yaw_rate"
                    ]
                )
            )
        )

        upright_values.append(
            float(
                info[
                    "upright"
                ]
            )
        )

        if (
            terminated
            or truncated
        ):
            break


    # ========================================================
    # SUMMARY
    # ========================================================

    end_position = (
        env.data.xpos[
            env.base_body_id
        ].copy()
    )

    displacement = float(
        np.linalg.norm(
            end_position[:2]
            - start_position[:2]
        )
    )

    mean_forward = float(
        np.mean(
            forward_values
        )
    )

    mean_abs_forward = float(
        np.mean(
            np.abs(
                forward_values
            )
        )
    )

    mean_abs_lateral = float(
        np.mean(
            np.abs(
                lateral_values
            )
        )
    )

    mean_abs_yaw = float(
        np.mean(
            np.abs(
                yaw_values
            )
        )
    )

    directional_ratio = (
        mean_abs_forward
        /
        (
            mean_abs_forward
            + mean_abs_lateral
            + 1e-9
        )
    )

    if path_length > 1e-9:

        path_efficiency = (
            displacement
            / path_length
        )

    else:

        path_efficiency = 0.0


    result = {

        "label":
            label,

        "time":
            np.asarray(
                times
            ),

        "x":
            np.asarray(
                x_values
            ),

        "y":
            np.asarray(
                y_values
            ),

        "forward":
            np.asarray(
                forward_values
            ),

        "lateral":
            np.asarray(
                lateral_values
            ),

        "yaw":
            np.asarray(
                yaw_values
            ),

        "mean_forward":
            mean_forward,

        "mean_abs_forward":
            mean_abs_forward,

        "mean_abs_lateral":
            mean_abs_lateral,

        "mean_abs_yaw":
            mean_abs_yaw,

        "directional_ratio":
            directional_ratio,

        "displacement":
            displacement,

        "path_length":
            path_length,

        "path_efficiency":
            path_efficiency,

        "min_upright":
            float(
                np.min(
                    upright_values
                )
            ),

        "terminated":
            terminated,
    }

    env.close()

    return result


# ============================================================
# 5. RUN BOTH
# ============================================================

A = rollout(
    MODEL_A,
    LABEL_A,
)

B = rollout(
    MODEL_B,
    LABEL_B,
)


# ============================================================
# 6. TERMINAL SUMMARY
# ============================================================

print()
print("=" * 72)
print("MODEL COMPARISON")
print("=" * 72)

print(
    f"{'Metric':<28}"
    f"{LABEL_A:>18}"
    f"{LABEL_B:>18}"
)

print("-" * 72)

print(
    f"{'Mean forward m/s':<28}"
    f"{A['mean_forward']:>18.5f}"
    f"{B['mean_forward']:>18.5f}"
)

print(
    f"{'Mean |forward| m/s':<28}"
    f"{A['mean_abs_forward']:>18.5f}"
    f"{B['mean_abs_forward']:>18.5f}"
)

print(
    f"{'Mean |lateral| m/s':<28}"
    f"{A['mean_abs_lateral']:>18.5f}"
    f"{B['mean_abs_lateral']:>18.5f}"
)

print(
    f"{'Mean |yaw| deg/s':<28}"
    f"{A['mean_abs_yaw']:>18.3f}"
    f"{B['mean_abs_yaw']:>18.3f}"
)

print(
    f"{'Directional ratio':<28}"
    f"{A['directional_ratio']:>18.4f}"
    f"{B['directional_ratio']:>18.4f}"
)

print(
    f"{'Path efficiency':<28}"
    f"{A['path_efficiency']:>18.4f}"
    f"{B['path_efficiency']:>18.4f}"
)

print(
    f"{'Displacement m':<28}"
    f"{A['displacement']:>18.4f}"
    f"{B['displacement']:>18.4f}"
)

print(
    f"{'Path length m':<28}"
    f"{A['path_length']:>18.4f}"
    f"{B['path_length']:>18.4f}"
)

print(
    f"{'Minimum upright':<28}"
    f"{A['min_upright']:>18.4f}"
    f"{B['min_upright']:>18.4f}"
)

print("=" * 72)


# ============================================================
# 7. SAVE SUMMARY CSV
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "04_summary.csv"
)

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "model",
        "mean_forward",
        "mean_abs_forward",
        "mean_abs_lateral",
        "mean_abs_yaw_deg_s",
        "directional_ratio",
        "displacement",
        "path_length",
        "path_efficiency",
        "min_upright",
        "terminated",
    ])

    for result in [
        A,
        B,
    ]:

        writer.writerow([
            result["label"],
            result["mean_forward"],
            result["mean_abs_forward"],
            result["mean_abs_lateral"],
            result["mean_abs_yaw"],
            result["directional_ratio"],
            result["displacement"],
            result["path_length"],
            result["path_efficiency"],
            result["min_upright"],
            result["terminated"],
        ])


# ============================================================
# 8. XY TRAJECTORY
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.plot(
    A["x"],
    A["y"],
    label=LABEL_A,
)

plt.plot(
    B["x"],
    B["y"],
    label=LABEL_B,
)

plt.scatter(
    [0],
    [0],
    marker="o",
    label="Start",
)

plt.xlabel(
    "World X displacement (m)"
)

plt.ylabel(
    "World Y displacement (m)"
)

plt.title(
    "Policy trajectory comparison"
)

plt.axis(
    "equal"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_trajectory.png",
    dpi=200,
)

plt.close()


# ============================================================
# 9. FORWARD VELOCITY
# ============================================================

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    A["time"],
    A["forward"],
    label=LABEL_A,
)

plt.plot(
    B["time"],
    B["forward"],
    label=LABEL_B,
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Forward velocity (m/s)"
)

plt.title(
    "Forward velocity"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_forward_velocity.png",
    dpi=200,
)

plt.close()


# ============================================================
# 10. LATERAL VELOCITY
# ============================================================

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    A["time"],
    A["lateral"],
    label=LABEL_A,
)

plt.plot(
    B["time"],
    B["lateral"],
    label=LABEL_B,
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Lateral velocity (m/s)"
)

plt.title(
    "Lateral velocity"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_lateral_velocity.png",
    dpi=200,
)

plt.close()


# ============================================================
# 11. YAW RATE
# ============================================================

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    A["time"],
    A["yaw"],
    label=LABEL_A,
)

plt.plot(
    B["time"],
    B["yaw"],
    label=LABEL_B,
)

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Yaw rate (deg/s)"
)

plt.title(
    "Yaw-rate comparison"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_yaw_rate.png",
    dpi=200,
)

plt.close()


# ============================================================
# 12. NORMALIZED METRIC COMPARISON
#
# Each metric has different units.
# Normalize B relative to A rather than putting raw
# m/s, deg/s and ratios on one y-axis.
# ============================================================

metric_names = [
    "Forward",
    "Lateral",
    "Yaw",
    "Directional",
    "Path efficiency",
]

a_metrics = np.array([
    A["mean_forward"],
    A["mean_abs_lateral"],
    A["mean_abs_yaw"],
    A["directional_ratio"],
    A["path_efficiency"],
])

b_metrics = np.array([
    B["mean_forward"],
    B["mean_abs_lateral"],
    B["mean_abs_yaw"],
    B["directional_ratio"],
    B["path_efficiency"],
])


# A = 100 %
a_normalized = np.ones(
    len(metric_names)
) * 100.0

b_normalized = (
    b_metrics
    /
    (
        np.abs(a_metrics)
        + 1e-9
    )
    * 100.0
)


x = np.arange(
    len(
        metric_names
    )
)

width = 0.35


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    x - width / 2,
    a_normalized,
    width,
    label=LABEL_A,
)

plt.bar(
    x + width / 2,
    b_normalized,
    width,
    label=LABEL_B,
)

plt.axhline(
    100,
    linewidth=1,
)

plt.xticks(
    x,
    metric_names,
    rotation=15,
)

plt.ylabel(
    f"Relative to {LABEL_A} (%)"
)

plt.title(
    "Normalized policy metrics"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_metrics.png",
    dpi=200,
)

plt.close()


# ============================================================
# DONE
# ============================================================

print()
print(
    "Comparison complete."
)

print(
    f"Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Generated:"
)

print(
    "  04_summary.csv"
)

print(
    "  04_trajectory.png"
)

print(
    "  04_forward_velocity.png"
)

print(
    "  04_lateral_velocity.png"
)

print(
    "  04_yaw_rate.png"
)

print(
    "  04_metrics.png"
)