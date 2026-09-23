#!/usr/bin/env python3

"""
04_compare_models.py
====================

Generic PPO / SAC / TD3 policy comparison tool.

Current:
- V2 PPO
- V3 SAC

Future:
- V4 TD3

IMPORTANT:
The original evaluation protocol is preserved.

Outputs:
- 04_trajectory.png
- 04_forward_velocity.png
- 04_lateral_velocity.png
- 04_yaw_rate.png
- 04_metrics.png
- 04_summary.csv

Raw reward is NOT compared because different reward functions
may use different scales.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3

from sesame_rl_env import (
    SesameRLEnv,
    CONTROL_DT,
    EPISODE_STEPS,
)


# ============================================================
# 1. MODELS
#
# 現在 V2 + V3 可以直接跑。
# 未來 V4 完成後，只要取消 V4 區塊註解。
# ============================================================

MODELS = [

    # --------------------------------------------------------
    # V2 — PPO
    # --------------------------------------------------------

    {
        "label": "V2 PPO",

        "algorithm": "PPO",

        "path": (
            "/home/kit/sesame-RL/"
            "003_rl_results/02_reward_v2_yaw/"
            "best_model/best_model.zip"
        ),
    },


    # --------------------------------------------------------
    # V3 — SAC
    # --------------------------------------------------------

    {
        "label": "V3 SAC",

        "algorithm": "SAC",

        "path": (
            "/home/kit/sesame-RL/"
            "003_rl_results/05_sac_v3/"
            "best_model/best_model.zip"
        ),
    },


    # --------------------------------------------------------
    # V4 — TD3
    #
    # V4 跑完後取消這段註解即可。
    # --------------------------------------------------------

    {
        "label": "V4 TD3",
    
        "algorithm": "TD3",
    
        "path": (
            "/home/kit/sesame-RL/"
            "003_rl_results/06_td3_v4/"
            "best_model/best_model.zip"
        ),
    },
]


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
#
# Same behaviour as first version:
# accept path with or without ".zip".
# ============================================================

def resolve_model_path(path):

    path = Path(path)


    if path.exists():

        return path


    zip_path = Path(
        str(path) + ".zip"
    )


    if zip_path.exists():

        return zip_path


    raise FileNotFoundError(
        f"Model not found:\n{path}"
    )


# ============================================================
# 4. ALGORITHM LOADER
#
# This is the main extension from the original version.
# ============================================================

LOADERS = {

    "PPO": PPO,

    "SAC": SAC,

    "TD3": TD3,
}


def load_model(
    algorithm,
    model_path,
):

    algorithm = algorithm.upper()


    if algorithm not in LOADERS:

        raise ValueError(
            f"Unsupported algorithm: {algorithm}\n"
            f"Supported: {list(LOADERS.keys())}"
        )


    ModelClass = LOADERS[
        algorithm
    ]


    # Keep original load behaviour.
    return ModelClass.load(
        str(model_path)
    )


# ============================================================
# 5. ROLLOUT
#
# Evaluation protocol below is kept from the first version.
# ============================================================

def rollout(
    model_path,
    label,
    algorithm,
):

    model_path = resolve_model_path(
        model_path
    )


    print()

    print(
        "=" * 65
    )


    print(
        f"Loading {label}"
    )


    print(
        f"Algorithm : {algorithm}"
    )


    print(
        model_path
    )


    print(
        "=" * 65
    )


    # --------------------------------------------------------
    # Correct algorithm loader
    # --------------------------------------------------------

    model = load_model(
        algorithm,
        model_path,
    )


    # --------------------------------------------------------
    # ORIGINAL EVALUATION PROTOCOL
    #
    # Important:
    # No w_yaw is passed here.
    # --------------------------------------------------------

    env = SesameRLEnv()


    obs, _ = env.reset(
        seed=SEED
    )


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    times = []


    x_values = []
    y_values = []


    forward_values = []
    lateral_values = []
    yaw_values = []


    upright_values = []


    # --------------------------------------------------------
    # ORIGINAL:
    # Base-body world position.
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Deterministic evaluation
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ORIGINAL:
        # Base-body world position using xpos.
        # ----------------------------------------------------

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
        # Time
        # ----------------------------------------------------

        times.append(
            step
            * CONTROL_DT
        )


        # ----------------------------------------------------
        # ORIGINAL:
        # Relative displacement from reset position.
        # ----------------------------------------------------

        x_values.append(
            position[0]
            - start_position[0]
        )


        y_values.append(
            position[1]
            - start_position[1]
        )


        # ----------------------------------------------------
        # Forward velocity
        # ----------------------------------------------------

        forward_values.append(
            float(
                info[
                    "forward_velocity"
                ]
            )
        )


        # ----------------------------------------------------
        # Lateral velocity
        # ----------------------------------------------------

        lateral_values.append(
            float(
                info[
                    "lateral_velocity"
                ]
            )
        )


        # ----------------------------------------------------
        # Yaw rate
        #
        # ORIGINAL:
        # Convert rad/s -> deg/s here.
        # ----------------------------------------------------

        yaw_values.append(
            float(
                np.rad2deg(
                    info[
                        "yaw_rate"
                    ]
                )
            )
        )


        # ----------------------------------------------------
        # Upright
        # ----------------------------------------------------

        upright_values.append(
            float(
                info[
                    "upright"
                ]
            )
        )


        # ----------------------------------------------------
        # End episode
        # ----------------------------------------------------

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


    # --------------------------------------------------------
    # Net displacement
    # --------------------------------------------------------

    displacement = float(
        np.linalg.norm(
            end_position[:2]
            - start_position[:2]
        )
    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Lateral
    # --------------------------------------------------------

    mean_abs_lateral = float(
        np.mean(
            np.abs(
                lateral_values
            )
        )
    )


    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    mean_abs_yaw = float(
        np.mean(
            np.abs(
                yaw_values
            )
        )
    )


    # --------------------------------------------------------
    # Directional ratio
    # --------------------------------------------------------

    directional_ratio = (
        mean_abs_forward
        /
        (
            mean_abs_forward
            + mean_abs_lateral
            + 1e-9
        )
    )


    # --------------------------------------------------------
    # Path efficiency
    # --------------------------------------------------------

    if path_length > 1e-9:

        path_efficiency = (
            displacement
            / path_length
        )

    else:

        path_efficiency = 0.0


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "label":
            label,

        "algorithm":
            algorithm,

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
# 6. RUN ALL ENABLED MODELS
# ============================================================

results = []


for config in MODELS:

    result = rollout(
        config[
            "path"
        ],

        config[
            "label"
        ],

        config[
            "algorithm"
        ],
    )


    results.append(
        result
    )


if len(results) < 2:

    raise RuntimeError(
        "At least two models are required for comparison."
    )


# ============================================================
# 7. TERMINAL SUMMARY
#
# Automatically supports 2, 3, or more models.
# ============================================================

print()

print(
    "=" * 110
)


print(
    "MODEL COMPARISON"
)


print(
    "=" * 110
)


header = (
    f"{'Metric':<28}"
)


for result in results:

    header += (
        f"{result['label']:>20}"
    )


print(
    header
)


print(
    "-" * 110
)


SUMMARY_ROWS = [

    (
        "Mean forward m/s",
        "mean_forward",
        ".5f",
    ),

    (
        "Mean |forward| m/s",
        "mean_abs_forward",
        ".5f",
    ),

    (
        "Mean |lateral| m/s",
        "mean_abs_lateral",
        ".5f",
    ),

    (
        "Mean |yaw| deg/s",
        "mean_abs_yaw",
        ".3f",
    ),

    (
        "Directional ratio",
        "directional_ratio",
        ".4f",
    ),

    (
        "Path efficiency",
        "path_efficiency",
        ".4f",
    ),

    (
        "Displacement m",
        "displacement",
        ".4f",
    ),

    (
        "Path length m",
        "path_length",
        ".4f",
    ),

    (
        "Minimum upright",
        "min_upright",
        ".4f",
    ),
]


for (
    metric_name,
    key,
    number_format,
) in SUMMARY_ROWS:

    row = (
        f"{metric_name:<28}"
    )


    for result in results:

        value = format(
            result[
                key
            ],
            number_format,
        )


        row += (
            f"{value:>20}"
        )


    print(
        row
    )


print(
    "=" * 110
)


# ============================================================
# 8. SAVE SUMMARY CSV
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
        "algorithm",
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


    for result in results:

        writer.writerow([
            result[
                "label"
            ],

            result[
                "algorithm"
            ],

            result[
                "mean_forward"
            ],

            result[
                "mean_abs_forward"
            ],

            result[
                "mean_abs_lateral"
            ],

            result[
                "mean_abs_yaw"
            ],

            result[
                "directional_ratio"
            ],

            result[
                "displacement"
            ],

            result[
                "path_length"
            ],

            result[
                "path_efficiency"
            ],

            result[
                "min_upright"
            ],

            result[
                "terminated"
            ],
        ])


# ============================================================
# 9. XY TRAJECTORY
#
# EXACT ORIGINAL:
# - base_body xpos
# - relative displacement
# - Start at (0,0)
# - same title
# - same axis labels
# - same figsize
# - same dpi
# ============================================================

plt.figure(
    figsize=(7, 7)
)


for result in results:

    plt.plot(
        result[
            "x"
        ],

        result[
            "y"
        ],

        label=result[
            "label"
        ],
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
# 10. FORWARD VELOCITY
# ============================================================

plt.figure(
    figsize=(10, 4)
)


for result in results:

    plt.plot(
        result[
            "time"
        ],

        result[
            "forward"
        ],

        label=result[
            "label"
        ],
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
# 11. LATERAL VELOCITY
# ============================================================

plt.figure(
    figsize=(10, 4)
)


for result in results:

    plt.plot(
        result[
            "time"
        ],

        result[
            "lateral"
        ],

        label=result[
            "label"
        ],
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
# 12. YAW RATE
# ============================================================

plt.figure(
    figsize=(10, 4)
)


for result in results:

    plt.plot(
        result[
            "time"
        ],

        result[
            "yaw"
        ],

        label=result[
            "label"
        ],
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
# 13. NORMALIZED POLICY METRICS
#
# ORIGINAL LOGIC:
# First model = baseline = 100 %
#
# V3/V4/etc. =
# metric / abs(baseline metric) * 100
# ============================================================

metric_names = [

    "Forward",

    "Lateral",

    "Yaw",

    "Directional",

    "Path efficiency",
]


baseline = results[0]


baseline_metrics = np.array([

    baseline[
        "mean_forward"
    ],

    baseline[
        "mean_abs_lateral"
    ],

    baseline[
        "mean_abs_yaw"
    ],

    baseline[
        "directional_ratio"
    ],

    baseline[
        "path_efficiency"
    ],
])


normalized_results = []


for index, result in enumerate(
    results
):

    # --------------------------------------------------------
    # Baseline = 100 %
    # --------------------------------------------------------

    if index == 0:

        normalized = (
            np.ones(
                len(
                    metric_names
                )
            )
            * 100.0
        )


    # --------------------------------------------------------
    # Other models relative to baseline
    # --------------------------------------------------------

    else:

        model_metrics = np.array([

            result[
                "mean_forward"
            ],

            result[
                "mean_abs_lateral"
            ],

            result[
                "mean_abs_yaw"
            ],

            result[
                "directional_ratio"
            ],

            result[
                "path_efficiency"
            ],
        ])


        normalized = (

            model_metrics

            /

            (
                np.abs(
                    baseline_metrics
                )

                + 1e-9
            )

            * 100.0
        )


    normalized_results.append(
        normalized
    )


# ------------------------------------------------------------
# Dynamic bar positions
#
# Works with:
# 2 models
# 3 models
# 4 models...
# ------------------------------------------------------------

x = np.arange(
    len(
        metric_names
    )
)


n_models = len(
    results
)


width = (
    0.8
    / n_models
)


offsets = (

    np.arange(
        n_models
    )

    - (
        n_models
        - 1
    ) / 2

) * width


plt.figure(
    figsize=(10, 5)
)


for (
    result,
    normalized,
    offset,
) in zip(

    results,

    normalized_results,

    offsets,
):

    plt.bar(

        x
        + offset,

        normalized,

        width,

        label=result[
            "label"
        ],
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
    f"Relative to {baseline['label']} (%)"
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
    "Output directory:"
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