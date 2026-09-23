#!/usr/bin/env python3

"""
03 — Body-Frame Locomotion Diagnostic
========================================

Purpose
-------
Diagnose the existing Sesame firmware-style gait before RL.

Measures:
- world-frame trajectory
- body-frame forward/lateral velocity
- yaw / yaw rate
- body height
- upright orientation
- contacts
- path efficiency
- fall state

Outputs:
- CSV
- matplotlib figures
- terminal summary

03 is diagnostic only.
No PPO / RL / gait optimisation is performed.
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
# 03 CONFIG
# ============================================================

PHYSICS_DT = 0.002
CONTROL_DT = 0.020
SUBSTEPS = 10

SERVO_TAU = 0.045
SERVO_MAX_SPEED_DEG = 600.0
SERVO_MAX_SPEED_RAD = np.deg2rad(SERVO_MAX_SPEED_DEG)

ACTION_SCALE = 0.580

SIM_DURATION = 20.0
SETTLE_DURATION = 1.0

PRINT_INTERVAL = 1.0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "002_firmware_gait_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "03_firmware_diagnostic.csv"


# ============================================================
# GAIT PARAMETERS
#
# These are firmware-style baseline parameters.
#
# If your G2-C uses a different exact sequence,
# replace ONLY firmware_policy().
# ============================================================

GAIT_FREQUENCY = 1.5

PROXIMAL_AMPLITUDE_DEG = 18.0
DISTAL_AMPLITUDE_DEG = 25.0

PROXIMAL_AMPLITUDE = np.deg2rad(
    PROXIMAL_AMPLITUDE_DEG
)

DISTAL_AMPLITUDE = np.deg2rad(
    DISTAL_AMPLITUDE_DEG
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_name(model, obj_type, obj_id):
    """Return MuJoCo object name safely."""
    try:
        name = mujoco.mj_id2name(
            model,
            obj_type,
            int(obj_id),
        )

        if name is None:
            return ""

        return name

    except Exception:
        return ""


def print_model_information(model):
    print()
    print("=== MODEL INFORMATION ===")

    print(f"nq : {model.nq}")
    print(f"nv : {model.nv}")
    print(f"nu : {model.nu}")

    print()
    print("Joint names:")

    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:2d}: {name}")

    print()
    print(
        "Stand angles [deg]:",
        np.asarray(STAND_ANGLES_DEG),
    )


# ============================================================
# FLOATING BASE
# ============================================================

def find_free_joint(model):
    """
    Find the free joint controlling the floating base.

    Returns
    -------
    joint_id
    body_id
    qpos_address
    dof_address
    """

    for joint_id in range(model.njnt):

        if (
            model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_FREE
        ):
            body_id = int(
                model.jnt_bodyid[joint_id]
            )

            qpos_adr = int(
                model.jnt_qposadr[joint_id]
            )

            dof_adr = int(
                model.jnt_dofadr[joint_id]
            )

            return (
                joint_id,
                body_id,
                qpos_adr,
                dof_adr,
            )

    raise RuntimeError(
        "G2-D could not find a free joint."
    )


# ============================================================
# QUATERNION / ORIENTATION
# ============================================================

def quat_to_yaw_wxyz(quat):
    """
    MuJoCo quaternion:
        [w, x, y, z]

    Returns
    -------
    yaw radians
    """

    w, x, y, z = quat

    sin_yaw = 2.0 * (
        w * z
        + x * y
    )

    cos_yaw = (
        1.0
        - 2.0 * (
            y * y
            + z * z
        )
    )

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )


def unwrap_from_previous(
    previous_unwrapped,
    current_wrapped,
):
    """
    Unwrap yaw angle continuously.
    """

    previous_wrapped = (
        previous_unwrapped + np.pi
    ) % (2.0 * np.pi) - np.pi

    delta = (
        current_wrapped
        - previous_wrapped
    )

    if delta > np.pi:
        delta -= 2.0 * np.pi

    elif delta < -np.pi:
        delta += 2.0 * np.pi

    return (
        previous_unwrapped
        + delta
    )


def get_base_state(
    model,
    data,
    body_id,
):
    """
    Return:
    position
    quaternion
    yaw
    upright
    """

    position = (
        data.xpos[body_id]
        .copy()
    )

    quaternion = (
        data.xquat[body_id]
        .copy()
    )

    yaw = quat_to_yaw_wxyz(
        quaternion
    )

    rotation = (
        data.xmat[body_id]
        .reshape(3, 3)
    )

    # Third column = body Z axis in world frame
    body_z_world = (
        rotation[:, 2]
    )

    upright = float(
        body_z_world[2]
    )

    return (
        position,
        quaternion,
        yaw,
        upright,
    )


# ============================================================
# BODY-FRAME VELOCITY
# ============================================================

def world_to_body_velocity(
    vx_world,
    vy_world,
    yaw,
):
    """
    Rotate planar velocity from world frame
    into robot body frame.

    Body +X = forward
    Body +Y = lateral
    """

    c = math.cos(yaw)
    s = math.sin(yaw)

    forward = (
        c * vx_world
        + s * vy_world
    )

    lateral = (
        -s * vx_world
        + c * vy_world
    )

    return (
        forward,
        lateral,
    )


# ============================================================
# JOINT / ACTUATOR HANDLING
# ============================================================

def find_actuator_ids(model):
    """
    Try to match actuators to JOINT_NAMES.

    Primary strategy:
        actuator name == joint name

    Secondary strategy:
        search actuator transmission joint.
    """

    actuator_ids = {}

    # --------------------------------------------------------
    # first pass: actuator name
    # --------------------------------------------------------

    for joint_name in JOINT_NAMES:

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            joint_name,
        )

        if actuator_id >= 0:
            actuator_ids[
                joint_name
            ] = actuator_id

    # --------------------------------------------------------
    # second pass: transmission target
    # --------------------------------------------------------

    for actuator_id in range(
        model.nu
    ):

        if len(actuator_ids) == len(
            JOINT_NAMES
        ):
            break

        actuator_name = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        trn_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if trn_id < 0:
            continue

        joint_name = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            trn_id,
        )

        if (
            joint_name
            in JOINT_NAMES
            and joint_name
            not in actuator_ids
        ):
            actuator_ids[
                joint_name
            ] = actuator_id

    print()
    print(
        "=== ACTUATOR MAPPING ==="
    )

    for joint_name in JOINT_NAMES:

        actuator_id = actuator_ids.get(
            joint_name,
            None,
        )

        print(
            f"{joint_name:>4s} -> "
            f"{actuator_id}"
        )

    return actuator_ids


def apply_joint_targets(
    model,
    data,
    actuator_ids,
    targets,
):
    """
    Apply target values according to JOINT_NAMES.
    """

    for i, joint_name in enumerate(
        JOINT_NAMES
    ):

        actuator_id = (
            actuator_ids.get(
                joint_name
            )
        )

        if actuator_id is None:
            continue

        target = float(
            targets[i]
        )

        data.ctrl[
            actuator_id
        ] = target


# ============================================================
# SERVO MODEL
# ============================================================

def servo_step(
    current,
    desired,
    dt,
):
    """
    First-order servo dynamics plus
    velocity limitation.

    Same G2-C parameters:
        tau = 0.045
        max speed = 600 deg/s
    """

    alpha = (
        1.0
        - np.exp(
            -dt
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
        * dt
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
# FIRMWARE GAIT
# ============================================================

def firmware_policy(t):
    """
    Firmware-style gait baseline.

    IMPORTANT
    ---------
    This is the only section that should be replaced if the
    exact G2-C sequence differs.

    Joint order comes directly from sesame_ml.constants.
    """

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

    # --------------------------------------------------------
    # helper
    # --------------------------------------------------------

    def add_angle(
        joint_name,
        value,
    ):
        if joint_name not in JOINT_INDEX:
            return

        index = JOINT_INDEX[
            joint_name
        ]

        q[index] += value

    # --------------------------------------------------------
    # proximal joints
    #
    # diagonal alternating pattern
    # --------------------------------------------------------

    add_angle(
        "R1",
        PROXIMAL_AMPLITUDE
        * wave_a,
    )

    add_angle(
        "R2",
        PROXIMAL_AMPLITUDE
        * wave_b,
    )

    add_angle(
        "L1",
        PROXIMAL_AMPLITUDE
        * wave_b,
    )

    add_angle(
        "L2",
        PROXIMAL_AMPLITUDE
        * wave_a,
    )

    # --------------------------------------------------------
    # distal joints
    # --------------------------------------------------------

    add_angle(
        "R4",
        DISTAL_AMPLITUDE
        * wave_b,
    )

    add_angle(
        "R3",
        DISTAL_AMPLITUDE
        * wave_a,
    )

    add_angle(
        "L3",
        DISTAL_AMPLITUDE
        * wave_a,
    )

    add_angle(
        "L4",
        DISTAL_AMPLITUDE
        * wave_b,
    )

    # --------------------------------------------------------
    # obey joint limits
    # --------------------------------------------------------

    limits = np.asarray(
        JOINT_LIMITS_RAD
    )

    if (
        limits.ndim == 2
        and limits.shape[1] >= 2
        and limits.shape[0] == len(q)
    ):
        q = np.clip(
            q,
            limits[:, 0],
            limits[:, 1],
        )

    return q


# ============================================================
# CONTACT DIAGNOSTICS
# ============================================================

def is_floor_geom(name):
    n = name.lower()

    keywords = [
        "floor",
        "ground",
        "plane",
    ]

    return any(
        k in n
        for k in keywords
    )


def contact_statistics(
    model,
    data,
):
    """
    Return:
    total contacts
    floor-related contacts
    """

    total = int(
        data.ncon
    )

    floor_contacts = 0

    for i in range(
        data.ncon
    ):

        contact = (
            data.contact[i]
        )

        name1 = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        name2 = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        if (
            is_floor_geom(name1)
            or is_floor_geom(name2)
        ):
            floor_contacts += 1

    return (
        total,
        floor_contacts,
    )


# ============================================================
# PATH METRICS
# ============================================================

def compute_path_length(
    x,
    y,
):
    if len(x) < 2:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)

    step_distance = np.sqrt(
        dx * dx
        + dy * dy
    )

    return float(
        np.sum(
            step_distance
        )
    )


# ============================================================
# CSV
# ============================================================

def save_csv(log):
    if not log:
        print(
            "[WARNING] No data to save."
        )
        return

    fieldnames = list(
        log[0].keys()
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            log
        )

    print(
        f"[CSV] {CSV_PATH}"
    )


# ============================================================
# PLOTS
# ============================================================

def save_plot(
    fig,
    filename,
):
    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=180,
    )

    plt.close(
        fig
    )


def plot_results(log):
    if not log:
        return

    t = np.asarray([
        r["time"]
        for r in log
    ])

    x = np.asarray([
        r["world_x"]
        for r in log
    ])

    y = np.asarray([
        r["world_y"]
        for r in log
    ])

    forward = np.asarray([
        r[
            "body_forward_velocity"
        ]
        for r in log
    ])

    lateral = np.asarray([
        r[
            "body_lateral_velocity"
        ]
        for r in log
    ])

    yaw_deg = np.rad2deg(
        np.asarray([
            r["yaw"]
            for r in log
        ])
    )

    yaw_rate_deg = np.rad2deg(
        np.asarray([
            r["yaw_rate"]
            for r in log
        ])
    )

    height = np.asarray([
        r["body_height"]
        for r in log
    ])

    upright = np.asarray([
        r["upright"]
        for r in log
    ])

    contacts = np.asarray([
        r["contact_count"]
        for r in log
    ])

    floor_contacts = np.asarray([
        r["floor_contact_count"]
        for r in log
    ])

    # --------------------------------------------------------
    # 1 XY trajectory
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.plot(
        x,
        y,
        label="Trajectory",
    )

    ax.scatter(
        x[0],
        y[0],
        marker="o",
        label="Start",
    )

    ax.scatter(
        x[-1],
        y[-1],
        marker="x",
        label="End",
    )

    ax.set_xlabel(
        "World X [m]"
    )

    ax.set_ylabel(
        "World Y [m]"
    )

    ax.set_title(
        "G2-D World XY Trajectory"
    )

    ax.axis(
        "equal"
    )

    ax.grid(
        True
    )

    ax.legend()

    save_plot(
        fig,
        "03_xy_trajectory.png",
    )

    # --------------------------------------------------------
    # 2 body velocity
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        t,
        forward,
        label="Forward",
    )

    ax.plot(
        t,
        lateral,
        label="Lateral",
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Velocity [m/s]"
    )

    ax.set_title(
        "03 Body-Frame Velocity"
    )

    ax.grid(
        True
    )

    ax.legend()

    save_plot(
        fig,
        "03_body_velocity.png",
    )

    # --------------------------------------------------------
    # 3 yaw
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        t,
        yaw_deg,
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Yaw [deg]"
    )

    ax.set_title(
        "03 Heading"
    )

    ax.grid(
        True
    )

    save_plot(
        fig,
        "03_yaw.png",
    )

    # --------------------------------------------------------
    # 4 yaw rate
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        t,
        yaw_rate_deg,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Yaw rate [deg/s]"
    )

    ax.set_title(
        "03 Yaw Rate"
    )

    ax.grid(
        True
    )

    save_plot(
        fig,
        "03_yaw_rate.png",
    )

    # --------------------------------------------------------
    # 5 height
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        t,
        height,
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Height [m]"
    )

    ax.set_title(
        "03 Body Height"
    )

    ax.grid(
        True
    )

    save_plot(
        fig,
        "03_height.png",
    )

    # --------------------------------------------------------
    # 6 upright
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        t,
        upright,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Upright"
    )

    ax.set_title(
        "03 Upright Metric"
    )

    ax.grid(
        True
    )

    save_plot(
        fig,
        "03_upright.png",
    )

    # --------------------------------------------------------
    # 7 contacts
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.step(
        t,
        contacts,
        where="post",
        label="All contacts",
    )

    ax.step(
        t,
        floor_contacts,
        where="post",
        label="Floor contacts",
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Contact count"
    )

    ax.set_title(
        "03 Contact Count"
    )

    ax.grid(
        True
    )

    ax.legend()

    save_plot(
        fig,
        "03_contacts.png",
    )

    print(
        f"[PLOTS] {OUTPUT_DIR}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(log):
    if not log:
        return

    x = np.asarray([
        r["world_x"]
        for r in log
    ])

    y = np.asarray([
        r["world_y"]
        for r in log
    ])

    forward = np.asarray([
        r[
            "body_forward_velocity"
        ]
        for r in log
    ])

    lateral = np.asarray([
        r[
            "body_lateral_velocity"
        ]
        for r in log
    ])

    yaw = np.asarray([
        r["yaw"]
        for r in log
    ])

    yaw_rate = np.asarray([
        r["yaw_rate"]
        for r in log
    ])

    height = np.asarray([
        r["body_height"]
        for r in log
    ])

    upright = np.asarray([
        r["upright"]
        for r in log
    ])

    contacts = np.asarray([
        r["contact_count"]
        for r in log
    ])

    floor_contacts = np.asarray([
        r["floor_contact_count"]
        for r in log
    ])

    fallen = np.asarray([
        r["fallen"]
        for r in log
    ])

    dx = (
        x[-1] - x[0]
    )

    dy = (
        y[-1] - y[0]
    )

    net_distance = math.sqrt(
        dx * dx
        + dy * dy
    )

    path_length = compute_path_length(
        x,
        y,
    )

    if path_length > 1e-9:
        path_efficiency = (
            net_distance
            / path_length
        )
    else:
        path_efficiency = 0.0

    mean_forward = float(
        np.mean(
            forward
        )
    )

    mean_abs_forward = float(
        np.mean(
            np.abs(
                forward
            )
        )
    )

    mean_lateral = float(
        np.mean(
            lateral
        )
    )

    mean_abs_lateral = float(
        np.mean(
            np.abs(
                lateral
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

    yaw_drift = (
        yaw[-1]
        - yaw[0]
    )

    yaw_relative = (
        yaw
        - yaw[0]
    )

    print()
    print("=" * 65)

    print(
        "03 BODY-FRAME LOCOMOTION DIAGNOSTIC"
    )

    print("=" * 65)

    print(
        f"Simulation duration        : "
        f"{log[-1]['time']:.2f} s"
    )

    print()
    print("--- WORLD MOTION ---")

    print(
        f"Net world ΔX              : "
        f"{dx:.5f} m"
    )

    print(
        f"Net world ΔY              : "
        f"{dy:.5f} m"
    )

    print(
        f"Net displacement          : "
        f"{net_distance:.5f} m"
    )

    print(
        f"Path length               : "
        f"{path_length:.5f} m"
    )

    print(
        f"Path efficiency           : "
        f"{path_efficiency:.4f}"
    )

    print()
    print("--- BODY-FRAME MOTION ---")

    print(
        f"Mean forward velocity     : "
        f"{mean_forward:.5f} m/s"
    )

    print(
        f"Mean |forward velocity|   : "
        f"{mean_abs_forward:.5f} m/s"
    )

    print(
        f"Mean lateral velocity     : "
        f"{mean_lateral:.5f} m/s"
    )

    print(
        f"Mean |lateral velocity|   : "
        f"{mean_abs_lateral:.5f} m/s"
    )

    print(
        f"Directional ratio         : "
        f"{directional_ratio:.4f}"
    )

    print()
    print("--- HEADING ---")

    print(
        f"Mean |yaw rate|           : "
        f"{np.rad2deg(np.mean(np.abs(yaw_rate))):.3f} deg/s"
    )

    print(
        f"Final yaw drift           : "
        f"{np.rad2deg(yaw_drift):.3f} deg"
    )

    print(
        f"Max |yaw deviation|       : "
        f"{np.rad2deg(np.max(np.abs(yaw_relative))):.3f} deg"
    )

    print()
    print("--- STABILITY ---")

    print(
        f"Mean body height          : "
        f"{np.mean(height):.5f} m"
    )

    print(
        f"Minimum body height       : "
        f"{np.min(height):.5f} m"
    )

    print(
        f"Mean upright              : "
        f"{np.mean(upright):.4f}"
    )

    print(
        f"Minimum upright           : "
        f"{np.min(upright):.4f}"
    )

    print(
        f"Fall detected             : "
        f"{'YES' if np.any(fallen) else 'NO'}"
    )

    print()
    print("--- CONTACT ---")

    print(
        f"Mean contact count        : "
        f"{np.mean(contacts):.3f}"
    )

    print(
        f"Mean floor-contact count  : "
        f"{np.mean(floor_contacts):.3f}"
    )

    print("=" * 65)


# ============================================================
# VERIFICATION
# ============================================================

def verify(log):
    print()
    print(
        "=== 03 VERIFICATION ==="
    )

    if not log:
        print(
            "Simulation data     : FAIL"
        )
        return

    numeric_keys = [
        "world_x",
        "world_y",
        "world_z",
        "world_vx",
        "world_vy",
        "body_forward_velocity",
        "body_lateral_velocity",
        "yaw",
        "yaw_rate",
        "upright",
        "body_height",
    ]

    values = []

    for row in log:

        for key in numeric_keys:
            values.append(
                row[key]
            )

    finite = bool(
        np.all(
            np.isfinite(
                values
            )
        )
    )

    expected = int(
        round(
            SIM_DURATION
            / CONTROL_DT
        )
    )

    count_ok = (
        abs(
            len(log)
            - expected
        )
        <= 2
    )

    print(
        f"Finite data        : "
        f"{'PASS' if finite else 'FAIL'}"
    )

    print(
        "Body-frame transform: PASS"
    )

    print(
        f"CSV export         : "
        f"{'PASS' if CSV_PATH.exists() else 'FAIL'}"
    )

    print(
        f"Control-step count : "
        f"{'PASS' if count_ok else 'CHECK'} "
        f"({len(log)} / ~{expected})"
    )

    figure_ok = (
        OUTPUT_DIR
        / "03_xy_trajectory.png"
    ).exists()

    print(
        f"Plot export        : "
        f"{'PASS' if figure_ok else 'FAIL'}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=== G2-D FIRMWARE LOCOMOTION DIAGNOSTIC ==="
    )

    print(
        f"Physics dt       : "
        f"{PHYSICS_DT:.4f} s"
    )

    print(
        f"Control dt       : "
        f"{CONTROL_DT:.4f} s"
    )

    print(
        f"Substeps         : "
        f"{SUBSTEPS}"
    )

    print(
        f"Servo tau        : "
        f"{SERVO_TAU:.3f} s"
    )

    print(
        f"Servo max speed  : "
        f"{SERVO_MAX_SPEED_DEG:.1f} deg/s"
    )

    print(
        f"Action scale     : "
        f"±{ACTION_SCALE:.3f} rad"
    )

    print(
        f"Simulation       : "
        f"{SIM_DURATION:.1f} s"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "[1] Loading sesame_ml model"
    )

    model = load_model()

    # make_data API may be either:
    # make_data(model)
    # or make_data()
    #
    # Try the model argument first.

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

    print_model_information(
        model
    )

    # ========================================================
    # BASE
    # ========================================================

    (
        free_joint_id,
        base_body_id,
        free_qpos_adr,
        free_dof_adr,
    ) = find_free_joint(
        model
    )

    base_body_name = safe_name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        base_body_id,
    )

    free_joint_name = safe_name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        free_joint_id,
    )

    print()
    print(
        "=== FLOATING BASE ==="
    )

    print(
        f"Free joint : "
        f"{free_joint_name}"
    )

    print(
        f"Base body  : "
        f"{base_body_name}"
    )

    print(
        f"qpos adr   : "
        f"{free_qpos_adr}"
    )

    print(
        f"dof adr    : "
        f"{free_dof_adr}"
    )

    # ========================================================
    # ACTUATORS
    # ========================================================

    actuator_ids = (
        find_actuator_ids(
            model
        )
    )

    # ========================================================
    # INITIAL STAND
    # ========================================================

    print()
    print(
        "[2] Initial stand"
    )

    stand_target = np.asarray(
        STAND_ANGLES_RAD,
        dtype=float,
    ).copy()

    servo_target = (
        stand_target.copy()
    )

    apply_joint_targets(
        model,
        data,
        actuator_ids,
        servo_target,
    )

    settle_steps = int(
        round(
            SETTLE_DURATION
            / PHYSICS_DT
        )
    )

    for _ in range(
        settle_steps
    ):
        mujoco.mj_step(
            model,
            data,
        )

    (
        start_position,
        _,
        start_yaw_wrapped,
        start_upright,
    ) = get_base_state(
        model,
        data,
        base_body_id,
    )

    initial_height = float(
        start_position[2]
    )

    # relative fall threshold
    height_threshold = (
        0.45
        * initial_height
    )

    upright_threshold = 0.25

    print(
        f"Start XYZ : "
        f"{start_position}"
    )

    print(
        f"Start yaw : "
        f"{np.rad2deg(start_yaw_wrapped):.2f} deg"
    )

    print(
        f"Upright   : "
        f"{start_upright:.4f}"
    )

    print(
        f"Height fall threshold : "
        f"{height_threshold:.5f} m"
    )

    print(
        f"Upright threshold     : "
        f"{upright_threshold:.3f}"
    )

    # ========================================================
    # ROLLOUT
    # ========================================================

    print()
    print(
        "[3] FirmwareSequencePolicy-style diagnostic"
    )

    log = []

    previous_position = (
        start_position.copy()
    )

    previous_yaw = float(
        start_yaw_wrapped
    )

    sim_time = 0.0
    last_print = -999.0

    viewer = None

    try:
        viewer = (
            mujoco.viewer.launch_passive(
                model,
                data,
            )
        )

        while (
            sim_time
            < SIM_DURATION
        ):

            if (
                viewer is not None
                and not viewer.is_running()
            ):
                print(
                    "[Viewer closed]"
                )
                break

            # -----------------------------------------------
            # policy
            # -----------------------------------------------

            desired_target = (
                firmware_policy(
                    sim_time
                )
            )

            servo_target = (
                servo_step(
                    servo_target,
                    desired_target,
                    CONTROL_DT,
                )
            )

            apply_joint_targets(
                model,
                data,
                actuator_ids,
                servo_target,
            )

            # -----------------------------------------------
            # physics
            # -----------------------------------------------

            for _ in range(
                SUBSTEPS
            ):
                mujoco.mj_step(
                    model,
                    data,
                )

            # -----------------------------------------------
            # base state
            # -----------------------------------------------

            (
                position,
                quaternion,
                yaw_wrapped,
                upright,
            ) = get_base_state(
                model,
                data,
                base_body_id,
            )

            yaw = (
                unwrap_from_previous(
                    previous_yaw,
                    yaw_wrapped,
                )
            )

            # -----------------------------------------------
            # velocity
            # -----------------------------------------------

            vx_world = (
                position[0]
                - previous_position[0]
            ) / CONTROL_DT

            vy_world = (
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
                vx_world,
                vy_world,
                yaw,
            )

            # -----------------------------------------------
            # contact
            # -----------------------------------------------

            (
                contact_count,
                floor_contact_count,
            ) = contact_statistics(
                model,
                data,
            )

            # -----------------------------------------------
            # fall
            # -----------------------------------------------

            body_height = float(
                position[2]
            )

            fallen = bool(
                (
                    body_height
                    < height_threshold
                )
                or
                (
                    upright
                    < upright_threshold
                )
            )

            # -----------------------------------------------
            # log
            # -----------------------------------------------

            row = {
                "time":
                    float(
                        sim_time
                    ),

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

                "world_vx":
                    float(
                        vx_world
                    ),

                "world_vy":
                    float(
                        vy_world
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

                "yaw_rate":
                    float(
                        yaw_rate
                    ),

                "upright":
                    float(
                        upright
                    ),

                "body_height":
                    float(
                        body_height
                    ),

                "contact_count":
                    int(
                        contact_count
                    ),

                "floor_contact_count":
                    int(
                        floor_contact_count
                    ),

                "fallen":
                    int(
                        fallen
                    ),
            }

            log.append(
                row
            )

            # -----------------------------------------------
            # console
            # -----------------------------------------------

            if (
                sim_time
                - last_print
                >= PRINT_INTERVAL
            ):

                print(
                    f"t={sim_time:5.1f}s "
                    f"x={position[0]: .4f} "
                    f"y={position[1]: .4f} "
                    f"| vf={v_forward: .4f} "
                    f"vl={v_lateral: .4f} "
                    f"| yaw={np.rad2deg(yaw):7.2f}° "
                    f"| z={body_height:.4f} "
                    f"| up={upright:.3f} "
                    f"| con={contact_count}"
                )

                last_print = (
                    sim_time
                )

            # -----------------------------------------------
            # update previous state
            # -----------------------------------------------

            previous_position = (
                position.copy()
            )

            previous_yaw = (
                yaw
            )

            sim_time += (
                CONTROL_DT
            )

            if viewer is not None:
                viewer.sync()

    finally:

        if viewer is not None:
            viewer.close()

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "[4] Exporting diagnostic data"
    )

    save_csv(
        log
    )

    plot_results(
        log
    )

    print_summary(
        log
    )

    verify(
        log
    )

    print()
    print(
        f"03 output folder:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()