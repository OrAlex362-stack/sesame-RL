#!/usr/bin/env python3

"""
04 — Firmware Gait Symmetry Diagnostic
========================================

Purpose
-------
Diagnose the source of lateral drift and yaw bias observed in 03.

03 result:
- stable posture
- no fall
- lateral velocity > forward velocity
- strong yaw oscillation / drift

04 therefore focuses on:
1. left/right joint tracking symmetry
2. commanded vs actual joint motion
3. left/right ground contact symmetry
4. per-gait-cycle forward/lateral/yaw displacement

No RL.
No gait optimisation.
"""

from pathlib import Path
import csv
import math

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

from sesame_ml.model import load_model, make_data

from sesame_ml.constants import (
    JOINT_INDEX,
    JOINT_NAMES,
    STAND_ANGLES_DEG,
    STAND_ANGLES_RAD,
    JOINT_LIMITS_RAD,
)


# ============================================================
# CONFIG
# ============================================================

PHYSICS_DT = 0.002
CONTROL_DT = 0.020
SUBSTEPS = 10

SERVO_TAU = 0.045
SERVO_MAX_SPEED_DEG = 600.0
SERVO_MAX_SPEED_RAD = np.deg2rad(SERVO_MAX_SPEED_DEG)

SIM_DURATION = 20.0
SETTLE_DURATION = 1.0

GAIT_FREQUENCY = 1.5
GAIT_PERIOD = 1.0 / GAIT_FREQUENCY

PROXIMAL_AMPLITUDE_DEG = 18.0
DISTAL_AMPLITUDE_DEG = 25.0

PROXIMAL_AMPLITUDE = np.deg2rad(
    PROXIMAL_AMPLITUDE_DEG
)

DISTAL_AMPLITUDE = np.deg2rad(
    DISTAL_AMPLITUDE_DEG
)

PRINT_INTERVAL = 1.0


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "002_firmware_gait_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEP_CSV = (
    OUTPUT_DIR
    / "04_step_diagnostic.csv"
)

CYCLE_CSV = (
    OUTPUT_DIR
    / "04_cycle_summary.csv"
)


# ============================================================
# LEG GROUPS
# ============================================================

RIGHT_JOINTS = [
    "R1",
    "R2",
    "R3",
    "R4",
]

LEFT_JOINTS = [
    "L1",
    "L2",
    "L3",
    "L4",
]


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_name(
    model,
    obj_type,
    obj_id,
):
    try:
        name = mujoco.mj_id2name(
            model,
            obj_type,
            int(obj_id),
        )

        return (
            name
            if name is not None
            else ""
        )

    except Exception:
        return ""


def find_free_joint(model):

    for joint_id in range(
        model.njnt
    ):

        if (
            model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_FREE
        ):

            body_id = int(
                model.jnt_bodyid[
                    joint_id
                ]
            )

            return (
                joint_id,
                body_id,
            )

    raise RuntimeError(
        "No floating-base free joint found."
    )


# ============================================================
# ORIENTATION
# ============================================================

def quat_to_yaw_wxyz(quat):

    w, x, y, z = quat

    sin_yaw = (
        2.0
        * (
            w * z
            + x * y
        )
    )

    cos_yaw = (
        1.0
        - 2.0
        * (
            y * y
            + z * z
        )
    )

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )


def unwrap_from_previous(
    previous,
    wrapped,
):

    previous_wrapped = (
        previous + np.pi
    ) % (2.0 * np.pi) - np.pi

    delta = (
        wrapped
        - previous_wrapped
    )

    if delta > np.pi:
        delta -= 2.0 * np.pi

    elif delta < -np.pi:
        delta += 2.0 * np.pi

    return (
        previous
        + delta
    )


def get_base_state(
    data,
    body_id,
):

    position = (
        data.xpos[
            body_id
        ].copy()
    )

    quat = (
        data.xquat[
            body_id
        ].copy()
    )

    yaw = quat_to_yaw_wxyz(
        quat
    )

    rotation = (
        data.xmat[
            body_id
        ].reshape(3, 3)
    )

    upright = float(
        rotation[2, 2]
    )

    return (
        position,
        yaw,
        upright,
    )


# ============================================================
# BODY-FRAME TRANSFORM
# ============================================================

def world_to_body_velocity(
    vx,
    vy,
    yaw,
):

    c = math.cos(yaw)
    s = math.sin(yaw)

    forward = (
        c * vx
        + s * vy
    )

    lateral = (
        -s * vx
        + c * vy
    )

    return (
        forward,
        lateral,
    )


def world_delta_to_body(
    dx,
    dy,
    yaw_reference,
):

    c = math.cos(
        yaw_reference
    )

    s = math.sin(
        yaw_reference
    )

    forward = (
        c * dx
        + s * dy
    )

    lateral = (
        -s * dx
        + c * dy
    )

    return (
        forward,
        lateral,
    )


# ============================================================
# JOINT MAPPING
# ============================================================

def build_joint_map(model):

    result = {}

    for name in JOINT_NAMES:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            print(
                f"[WARNING] joint not found: {name}"
            )
            continue

        result[name] = {
            "joint_id":
                joint_id,

            "qpos_adr":
                int(
                    model.jnt_qposadr[
                        joint_id
                    ]
                ),
        }

    return result


def build_actuator_map(model):

    result = {}

    for name in JOINT_NAMES:

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if actuator_id >= 0:

            result[
                name
            ] = actuator_id

            continue

        # fallback via transmission joint
        for aid in range(
            model.nu
        ):

            joint_id = int(
                model.actuator_trnid[
                    aid,
                    0,
                ]
            )

            if joint_id < 0:
                continue

            target_name = safe_name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if target_name == name:

                result[
                    name
                ] = aid

                break

    print()
    print(
        "=== ACTUATOR MAP ==="
    )

    for name in JOINT_NAMES:

        print(
            f"{name:>3s} -> "
            f"{result.get(name)}"
        )

    return result


# ============================================================
# JOINT READ / CONTROL
# ============================================================

def read_joint_positions(
    data,
    joint_map,
):

    q = {}

    for name in JOINT_NAMES:

        info = joint_map.get(
            name
        )

        if info is None:
            q[name] = np.nan
            continue

        q[name] = float(
            data.qpos[
                info["qpos_adr"]
            ]
        )

    return q


def apply_joint_targets(
    data,
    actuator_map,
    targets,
):

    for name in JOINT_NAMES:

        actuator_id = (
            actuator_map.get(
                name
            )
        )

        if actuator_id is None:
            continue

        index = JOINT_INDEX[
            name
        ]

        data.ctrl[
            actuator_id
        ] = targets[
            index
        ]


# ============================================================
# SERVO
# ============================================================

def servo_step(
    current,
    desired,
):

    alpha = (
        1.0
        - np.exp(
            -CONTROL_DT
            / SERVO_TAU
        )
    )

    filtered = (
        current
        + alpha
        * (
            desired
            - current
        )
    )

    max_delta = (
        SERVO_MAX_SPEED_RAD
        * CONTROL_DT
    )

    delta = np.clip(
        filtered - current,
        -max_delta,
        max_delta,
    )

    return (
        current
        + delta
    )


# ============================================================
# SAME GAIT AS CURRENT 03
# ============================================================

def firmware_policy(t):

    q = np.array(
        STAND_ANGLES_RAD,
        dtype=float,
        copy=True,
    )

    phase = (
        2.0
        * np.pi
        * GAIT_FREQUENCY
        * t
    )

    wave_a = math.sin(
        phase
    )

    wave_b = math.sin(
        phase + np.pi
    )

    def add(
        name,
        value,
    ):

        if name in JOINT_INDEX:

            q[
                JOINT_INDEX[
                    name
                ]
            ] += value

    # proximal
    add(
        "R1",
        PROXIMAL_AMPLITUDE
        * wave_a,
    )

    add(
        "R2",
        PROXIMAL_AMPLITUDE
        * wave_b,
    )

    add(
        "L1",
        PROXIMAL_AMPLITUDE
        * wave_b,
    )

    add(
        "L2",
        PROXIMAL_AMPLITUDE
        * wave_a,
    )

    # distal
    add(
        "R4",
        DISTAL_AMPLITUDE
        * wave_b,
    )

    add(
        "R3",
        DISTAL_AMPLITUDE
        * wave_a,
    )

    add(
        "L3",
        DISTAL_AMPLITUDE
        * wave_a,
    )

    add(
        "L4",
        DISTAL_AMPLITUDE
        * wave_b,
    )

    limits = np.asarray(
        JOINT_LIMITS_RAD
    )

    if (
        limits.ndim == 2
        and limits.shape[0]
        == len(q)
    ):

        q = np.clip(
            q,
            limits[:, 0],
            limits[:, 1],
        )

    return q


# ============================================================
# CONTACT CLASSIFICATION
# ============================================================

def classify_side_from_name(name):

    n = name.lower()

    if (
        n.startswith("r")
        or "right" in n
        or "_r" in n
    ):
        return "R"

    if (
        n.startswith("l")
        or "left" in n
        or "_l" in n
    ):
        return "L"

    return None


def geom_side(
    model,
    geom_id,
):
    """
    Determine left/right side using:
    1. geom name
    2. geom body name
    """

    geom_name = safe_name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )

    side = classify_side_from_name(
        geom_name
    )

    if side is not None:
        return side

    body_id = int(
        model.geom_bodyid[
            geom_id
        ]
    )

    body_name = safe_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    return classify_side_from_name(
        body_name
    )


def is_ground(
    model,
    geom_id,
):

    name = safe_name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    ).lower()

    return any(
        key in name
        for key in [
            "floor",
            "ground",
            "plane",
        ]
    )


def get_side_contacts(
    model,
    data,
):

    right = 0
    left = 0

    total_ground = 0

    for i in range(
        data.ncon
    ):

        con = data.contact[i]

        g1 = int(
            con.geom1
        )

        g2 = int(
            con.geom2
        )

        ground1 = is_ground(
            model,
            g1,
        )

        ground2 = is_ground(
            model,
            g2,
        )

        if not (
            ground1
            or ground2
        ):
            continue

        total_ground += 1

        robot_geom = (
            g2
            if ground1
            else g1
        )

        side = geom_side(
            model,
            robot_geom,
        )

        if side == "R":
            right += 1

        elif side == "L":
            left += 1

    return (
        right,
        left,
        total_ground,
    )


# ============================================================
# ERROR METRICS
# ============================================================

def side_tracking_error(
    command,
    actual,
    side_joint_names,
):

    errors = []

    for name in side_joint_names:

        if name not in JOINT_INDEX:
            continue

        q_actual = actual.get(
            name,
            np.nan,
        )

        if not np.isfinite(
            q_actual
        ):
            continue

        q_command = command[
            JOINT_INDEX[
                name
            ]
        ]

        errors.append(
            abs(
                q_command
                - q_actual
            )
        )

    if not errors:
        return np.nan

    return float(
        np.mean(
            errors
        )
    )


# ============================================================
# CSV
# ============================================================

def save_csv(
    path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# CYCLE ANALYSIS
# ============================================================

def analyse_cycles(log):

    if not log:
        return []

    cycles = []

    cycle_ids = sorted(
        set(
            row["cycle"]
            for row in log
        )
    )

    for cycle_id in cycle_ids:

        rows = [
            row
            for row in log
            if row["cycle"]
            == cycle_id
        ]

        # ignore partial cycle at end
        expected = int(
            round(
                GAIT_PERIOD
                / CONTROL_DT
            )
        )

        if len(rows) < (
            expected * 0.8
        ):
            continue

        start = rows[0]
        end = rows[-1]

        dx = (
            end["world_x"]
            - start["world_x"]
        )

        dy = (
            end["world_y"]
            - start["world_y"]
        )

        forward_delta, lateral_delta = (
            world_delta_to_body(
                dx,
                dy,
                start["yaw"],
            )
        )

        yaw_delta = (
            end["yaw"]
            - start["yaw"]
        )

        right_contact_duty = np.mean([
            row["right_contact"]
            > 0
            for row in rows
        ])

        left_contact_duty = np.mean([
            row["left_contact"]
            > 0
            for row in rows
        ])

        right_error = np.nanmean([
            row[
                "right_tracking_error"
            ]
            for row in rows
        ])

        left_error = np.nanmean([
            row[
                "left_tracking_error"
            ]
            for row in rows
        ])

        cycles.append({

            "cycle":
                cycle_id,

            "start_time":
                start["time"],

            "end_time":
                end["time"],

            "delta_forward":
                forward_delta,

            "delta_lateral":
                lateral_delta,

            "delta_yaw_deg":
                np.rad2deg(
                    yaw_delta
                ),

            "right_contact_duty":
                right_contact_duty,

            "left_contact_duty":
                left_contact_duty,

            "contact_duty_difference":
                (
                    right_contact_duty
                    - left_contact_duty
                ),

            "right_tracking_error_deg":
                np.rad2deg(
                    right_error
                ),

            "left_tracking_error_deg":
                np.rad2deg(
                    left_error
                ),

            "tracking_error_difference_deg":
                np.rad2deg(
                    right_error
                    - left_error
                ),
        })

    return cycles


# ============================================================
# PLOTTING
# ============================================================

def save_figure(
    fig,
    filename,
):

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / filename,
        dpi=180,
    )

    plt.close(fig)


def plot_step_results(log):

    time = np.asarray([
        r["time"]
        for r in log
    ])

    # --------------------------------------------------------
    # joint tracking
    # --------------------------------------------------------

    for joint_name in JOINT_NAMES:

        cmd_key = (
            f"{joint_name}_cmd"
        )

        actual_key = (
            f"{joint_name}_actual"
        )

        if (
            cmd_key
            not in log[0]
        ):
            continue

        command = np.rad2deg(
            np.asarray([
                r[cmd_key]
                for r in log
            ])
        )

        actual = np.rad2deg(
            np.asarray([
                r[actual_key]
                for r in log
            ])
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            time,
            command,
            label="Command",
        )

        ax.plot(
            time,
            actual,
            label="Actual",
        )

        ax.set_xlabel(
            "Time [s]"
        )

        ax.set_ylabel(
            "Joint angle [deg]"
        )

        ax.set_title(
            f"04-{joint_name} Tracking"
        )

        ax.grid(True)
        ax.legend()

        save_figure(
            fig,
            f"04_{joint_name}_tracking.png",
        )

    # --------------------------------------------------------
    # left/right tracking error
    # --------------------------------------------------------

    right_error = np.rad2deg(
        np.asarray([
            r[
                "right_tracking_error"
            ]
            for r in log
        ])
    )

    left_error = np.rad2deg(
        np.asarray([
            r[
                "left_tracking_error"
            ]
            for r in log
        ])
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        time,
        right_error,
        label="Right",
    )

    ax.plot(
        time,
        left_error,
        label="Left",
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Mean |tracking error| [deg]"
    )

    ax.set_title(
        "04- Left / Right Tracking Error"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "04_side_tracking_error.png",
    )

    # --------------------------------------------------------
    # contacts
    # --------------------------------------------------------

    right_contact = np.asarray([
        r["right_contact"]
        for r in log
    ])

    left_contact = np.asarray([
        r["left_contact"]
        for r in log
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.step(
        time,
        right_contact,
        where="post",
        label="Right",
    )

    ax.step(
        time,
        left_contact,
        where="post",
        label="Left",
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Ground contact count"
    )

    ax.set_title(
        "04- Left / Right Ground Contacts"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "04_side_contacts.png",
    )


def plot_cycles(cycles):

    if not cycles:
        return

    cycle = np.asarray([
        r["cycle"]
        for r in cycles
    ])

    forward = np.asarray([
        r["delta_forward"]
        for r in cycles
    ])

    lateral = np.asarray([
        r["delta_lateral"]
        for r in cycles
    ])

    yaw = np.asarray([
        r["delta_yaw_deg"]
        for r in cycles
    ])

    # displacement
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        cycle,
        forward,
        marker="o",
        label="Δ forward",
    )

    ax.plot(
        cycle,
        lateral,
        marker="o",
        label="Δ lateral",
    )

    ax.axhline(
        0.0
    )

    ax.set_xlabel(
        "Gait cycle"
    )

    ax.set_ylabel(
        "Displacement [m]"
    )

    ax.set_title(
        "04- Per-Cycle Body Displacement"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "04_cycle_displacement.png",
    )

    # yaw
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        cycle,
        yaw,
        marker="o",
    )

    ax.axhline(
        0.0
    )

    ax.set_xlabel(
        "Gait cycle"
    )

    ax.set_ylabel(
        "Δ yaw [deg]"
    )

    ax.set_title(
        "04- Per-Cycle Yaw Bias"
    )

    ax.grid(True)

    save_figure(
        fig,
        "04_cycle_yaw.png",
    )

    # contact duty
    right = np.asarray([
        r["right_contact_duty"]
        for r in cycles
    ])

    left = np.asarray([
        r["left_contact_duty"]
        for r in cycles
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        cycle,
        right,
        marker="o",
        label="Right",
    )

    ax.plot(
        cycle,
        left,
        marker="o",
        label="Left",
    )

    ax.set_xlabel(
        "Gait cycle"
    )

    ax.set_ylabel(
        "Contact duty factor"
    )

    ax.set_title(
        "04- Left / Right Contact Duty"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "04_contact_duty.png",
    )


# ============================================================
# SUMMARY
# ============================================================

def print_cycle_summary(
    cycles,
):

    print()
    print("=" * 68)
    print(
        "04 GAIT SYMMETRY DIAGNOSTIC"
    )
    print("=" * 68)

    if not cycles:

        print(
            "No complete gait cycles found."
        )
        return

    forward = np.asarray([
        r["delta_forward"]
        for r in cycles
    ])

    lateral = np.asarray([
        r["delta_lateral"]
        for r in cycles
    ])

    yaw = np.asarray([
        r["delta_yaw_deg"]
        for r in cycles
    ])

    right_duty = np.asarray([
        r["right_contact_duty"]
        for r in cycles
    ])

    left_duty = np.asarray([
        r["left_contact_duty"]
        for r in cycles
    ])

    right_error = np.asarray([
        r[
            "right_tracking_error_deg"
        ]
        for r in cycles
    ])

    left_error = np.asarray([
        r[
            "left_tracking_error_deg"
        ]
        for r in cycles
    ])

    print(
        f"Complete cycles             : "
        f"{len(cycles)}"
    )

    print()
    print(
        "--- PER-CYCLE MOTION ---"
    )

    print(
        f"Mean Δforward / cycle       : "
        f"{np.mean(forward): .5f} m"
    )

    print(
        f"Mean Δlateral / cycle       : "
        f"{np.mean(lateral): .5f} m"
    )

    print(
        f"Mean |Δlateral| / cycle     : "
        f"{np.mean(np.abs(lateral)): .5f} m"
    )

    print(
        f"Mean Δyaw / cycle           : "
        f"{np.mean(yaw): .3f} deg"
    )

    print(
        f"Mean |Δyaw| / cycle         : "
        f"{np.mean(np.abs(yaw)): .3f} deg"
    )

    print()
    print(
        "--- CONTACT SYMMETRY ---"
    )

    print(
        f"Right contact duty          : "
        f"{np.mean(right_duty):.4f}"
    )

    print(
        f"Left contact duty           : "
        f"{np.mean(left_duty):.4f}"
    )

    print(
        f"R-L duty difference         : "
        f"{np.mean(right_duty-left_duty):+.4f}"
    )

    print()
    print(
        "--- JOINT TRACKING ---"
    )

    print(
        f"Right tracking error        : "
        f"{np.nanmean(right_error):.3f} deg"
    )

    print(
        f"Left tracking error         : "
        f"{np.nanmean(left_error):.3f} deg"
    )

    print(
        f"R-L tracking difference     : "
        f"{np.nanmean(right_error-left_error):+.3f} deg"
    )

    print()
    print(
        "--- CONSISTENCY ---"
    )

    positive_yaw = np.mean(
        yaw > 0
    )

    negative_yaw = np.mean(
        yaw < 0
    )

    positive_lat = np.mean(
        lateral > 0
    )

    negative_lat = np.mean(
        lateral < 0
    )

    print(
        f"Cycles with +yaw            : "
        f"{positive_yaw:.3f}"
    )

    print(
        f"Cycles with -yaw            : "
        f"{negative_yaw:.3f}"
    )

    print(
        f"Cycles with +lateral        : "
        f"{positive_lat:.3f}"
    )

    print(
        f"Cycles with -lateral        : "
        f"{negative_lat:.3f}"
    )

    print("=" * 68)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=== 04 — FIRMWARE GAIT SYMMETRY DIAGNOSTIC ==="
    )

    print(
        f"Physics dt       : {PHYSICS_DT:.4f} s"
    )

    print(
        f"Control dt       : {CONTROL_DT:.4f} s"
    )

    print(
        f"Substeps         : {SUBSTEPS}"
    )

    print(
        f"Gait frequency   : {GAIT_FREQUENCY:.3f} Hz"
    )

    print(
        f"Gait period      : {GAIT_PERIOD:.4f} s"
    )

    print(
        f"Simulation       : {SIM_DURATION:.1f} s"
    )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = load_model()

    try:
        data = make_data(
            model
        )

    except TypeError:
        data = make_data()

    model.opt.timestep = (
        PHYSICS_DT
    )

    mujoco.mj_forward(
        model,
        data,
    )

    # --------------------------------------------------------
    # mappings
    # --------------------------------------------------------

    (
        free_joint_id,
        base_body_id,
    ) = find_free_joint(
        model
    )

    base_name = safe_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_id,
    )

    print(
        f"Base body        : {base_name}"
    )

    joint_map = build_joint_map(
        model
    )

    actuator_map = (
        build_actuator_map(
            model
        )
    )

    # --------------------------------------------------------
    # stand
    # --------------------------------------------------------

    stand = np.asarray(
        STAND_ANGLES_RAD,
        dtype=float,
    ).copy()

    servo_target = (
        stand.copy()
    )

    apply_joint_targets(
        data,
        actuator_map,
        servo_target,
    )

    settle_steps = int(
        round(
            SETTLE_DURATION
            / PHYSICS_DT
        )
    )

    print()
    print(
        "[1] Initial stand"
    )

    for _ in range(
        settle_steps
    ):
        mujoco.mj_step(
            model,
            data,
        )

    (
        previous_position,
        start_yaw,
        start_upright,
    ) = get_base_state(
        data,
        base_body_id,
    )

    previous_yaw = (
        start_yaw
    )

    print(
        f"Start XYZ : {previous_position}"
    )

    print(
        f"Start yaw : "
        f"{np.rad2deg(start_yaw):.2f} deg"
    )

    print(
        f"Upright   : {start_upright:.4f}"
    )

    # --------------------------------------------------------
    # FIXED CONTROL STEP COUNT
    # --------------------------------------------------------

    num_steps = int(
        round(
            SIM_DURATION
            / CONTROL_DT
        )
    )

    print()
    print(
        f"[2] Running {num_steps} fixed control steps"
    )

    log = []

    last_print = -1.0

    viewer = (
        mujoco.viewer.launch_passive(
            model,
            data,
        )
    )

    try:

        for step in range(
            num_steps
        ):

            if not viewer.is_running():
                print(
                    "[Viewer closed]"
                )
                break

            sim_time = (
                step
                * CONTROL_DT
            )

            cycle_id = int(
                math.floor(
                    sim_time
                    / GAIT_PERIOD
                )
            )

            phase = (
                (
                    sim_time
                    % GAIT_PERIOD
                )
                / GAIT_PERIOD
            )

            # ------------------------------------------------
            # command
            # ------------------------------------------------

            desired = (
                firmware_policy(
                    sim_time
                )
            )

            servo_target = (
                servo_step(
                    servo_target,
                    desired,
                )
            )

            apply_joint_targets(
                data,
                actuator_map,
                servo_target,
            )

            # ------------------------------------------------
            # physics
            # ------------------------------------------------

            for _ in range(
                SUBSTEPS
            ):

                mujoco.mj_step(
                    model,
                    data,
                )

            # ------------------------------------------------
            # actual joint state
            # ------------------------------------------------

            actual = (
                read_joint_positions(
                    data,
                    joint_map,
                )
            )

            right_error = (
                side_tracking_error(
                    servo_target,
                    actual,
                    RIGHT_JOINTS,
                )
            )

            left_error = (
                side_tracking_error(
                    servo_target,
                    actual,
                    LEFT_JOINTS,
                )
            )

            # ------------------------------------------------
            # base state
            # ------------------------------------------------

            (
                position,
                wrapped_yaw,
                upright,
            ) = get_base_state(
                data,
                base_body_id,
            )

            yaw = (
                unwrap_from_previous(
                    previous_yaw,
                    wrapped_yaw,
                )
            )

            vx = (
                position[0]
                - previous_position[0]
            ) / CONTROL_DT

            vy = (
                position[1]
                - previous_position[1]
            ) / CONTROL_DT

            yaw_rate = (
                yaw
                - previous_yaw
            ) / CONTROL_DT

            (
                v_forward,
                v_lateral,
            ) = world_to_body_velocity(
                vx,
                vy,
                yaw,
            )

            # ------------------------------------------------
            # contacts
            # ------------------------------------------------

            (
                right_contact,
                left_contact,
                floor_contacts,
            ) = get_side_contacts(
                model,
                data,
            )

            # ------------------------------------------------
            # base row
            # ------------------------------------------------

            row = {

                "step":
                    step,

                "time":
                    sim_time,

                "cycle":
                    cycle_id,

                "phase":
                    phase,

                "world_x":
                    float(
                        position[0]
                    ),

                "world_y":
                    float(
                        position[1]
                    ),

                "world_z":
                    float(
                        position[2]
                    ),

                "body_forward_velocity":
                    float(
                        v_forward
                    ),

                "body_lateral_velocity":
                    float(
                        v_lateral
                    ),

                "yaw":
                    float(
                        yaw
                    ),

                "yaw_deg":
                    float(
                        np.rad2deg(
                            yaw
                        )
                    ),

                "yaw_rate_deg_s":
                    float(
                        np.rad2deg(
                            yaw_rate
                        )
                    ),

                "upright":
                    float(
                        upright
                    ),

                "right_contact":
                    int(
                        right_contact
                    ),

                "left_contact":
                    int(
                        left_contact
                    ),

                "floor_contact_count":
                    int(
                        floor_contacts
                    ),

                "right_tracking_error":
                    float(
                        right_error
                    ),

                "left_tracking_error":
                    float(
                        left_error
                    ),
            }

            # ------------------------------------------------
            # add all joint commands / actuals
            # ------------------------------------------------

            for name in JOINT_NAMES:

                index = JOINT_INDEX[
                    name
                ]

                row[
                    f"{name}_cmd"
                ] = float(
                    servo_target[
                        index
                    ]
                )

                row[
                    f"{name}_actual"
                ] = float(
                    actual.get(
                        name,
                        np.nan,
                    )
                )

                row[
                    f"{name}_error"
                ] = float(
                    servo_target[
                        index
                    ]
                    - actual.get(
                        name,
                        np.nan,
                    )
                )

            log.append(
                row
            )

            # ------------------------------------------------
            # console
            # ------------------------------------------------

            if (
                sim_time
                - last_print
                >= PRINT_INTERVAL
            ):

                print(
                    f"t={sim_time:5.1f}s "
                    f"cycle={cycle_id:02d} "
                    f"phase={phase:.2f} "
                    f"| vf={v_forward:+.4f} "
                    f"vl={v_lateral:+.4f} "
                    f"| yaw={np.rad2deg(yaw):+7.2f}° "
                    f"| Rc={right_contact} "
                    f"Lc={left_contact} "
                    f"| Rerr={np.rad2deg(right_error):.2f}° "
                    f"Lerr={np.rad2deg(left_error):.2f}°"
                )

                last_print = (
                    sim_time
                )

            previous_position = (
                position.copy()
            )

            previous_yaw = (
                yaw
            )

            viewer.sync()

    finally:

        viewer.close()

    # --------------------------------------------------------
    # export step data
    # --------------------------------------------------------

    save_csv(
        STEP_CSV,
        log,
    )

    # --------------------------------------------------------
    # cycle analysis
    # --------------------------------------------------------

    cycles = analyse_cycles(
        log
    )

    save_csv(
        CYCLE_CSV,
        cycles,
    )

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------

    plot_step_results(
        log
    )

    plot_cycles(
        cycles
    )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    print_cycle_summary(
        cycles
    )

    print()
    print(
        "=== 04 VERIFICATION ==="
    )

    print(
        f"Control steps      : "
        f"{len(log)} / {num_steps}"
    )

    print(
        f"Complete cycles    : "
        f"{len(cycles)}"
    )

    print(
        f"Step CSV           : "
        f"{'PASS' if STEP_CSV.exists() else 'FAIL'}"
    )

    print(
        f"Cycle CSV          : "
        f"{'PASS' if CYCLE_CSV.exists() else 'FAIL'}"
    )

    print(
        f"Output folder      : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()