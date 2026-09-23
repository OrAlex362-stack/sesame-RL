#!/usr/bin/env python3

"""
05 / G2-F — Per-Leg Contact & Phase Diagnostic
===============================================

Goal
----
Explain the systematic lateral drift found in G2-D / G2-E.

G2-E findings:
- Mean Δforward / cycle  ≈ -0.00080 m
- Mean Δlateral / cycle  ≈ +0.01624 m
- Mean Δyaw / cycle      ≈ +2.076 deg
- Right contact duty     ≈ 0.510
- Left contact duty      ≈ 0.416
- Joint tracking asymmetry is small

This experiment asks:

1. Which individual leg has abnormal contact duty?
2. At what gait phase does touchdown / liftoff occur?
3. Which leg contacts coincide with lateral velocity?
4. Which leg contacts coincide with yaw motion?
5. Is the lateral bias generated repeatedly at the same phase?
6. Is the asymmetry contact-related rather than actuator-tracking-related?

NO RL.
NO gait tuning.
NO amplitude correction.

The controller must remain identical to G2-E.
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
# 1. EXPERIMENT CONFIG
# ============================================================

PHYSICS_DT = 0.002
CONTROL_DT = 0.020
SUBSTEPS = 10

SERVO_TAU = 0.045

SERVO_MAX_SPEED_DEG = 600.0
SERVO_MAX_SPEED_RAD = np.deg2rad(
    SERVO_MAX_SPEED_DEG
)

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

# Number of normalized gait-phase bins
PHASE_BINS = 20


# ============================================================
# 2. OUTPUT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Keep all firmware diagnostic results together
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
    / "05_step_diagnostic.csv"
)

CYCLE_CSV = (
    OUTPUT_DIR
    / "05_cycle_summary.csv"
)

PHASE_CSV = (
    OUTPUT_DIR
    / "05_phase_summary.csv"
)

EVENT_CSV = (
    OUTPUT_DIR
    / "05_contact_events.csv"
)

GEOM_MAP_CSV = (
    OUTPUT_DIR
    / "05_geom_leg_map.csv"
)


# ============================================================
# 3. LEG DEFINITIONS
# ============================================================

"""
Contact attribution is NOT based only on geom names.

Each leg is identified from its proximal joint body and its
descendant bodies in the MuJoCo kinematic tree.

Expected Sesame topology:

R1 -> R4
R2 -> R3
L1 -> L3
L2 -> L4
"""

LEG_DEFS = {
    "R1-R4": {
        "root_joint": "R1",
        "joints": ["R1", "R4"],
        "side": "R",
    },

    "R2-R3": {
        "root_joint": "R2",
        "joints": ["R2", "R3"],
        "side": "R",
    },

    "L1-L3": {
        "root_joint": "L1",
        "joints": ["L1", "L3"],
        "side": "L",
    },

    "L2-L4": {
        "root_joint": "L2",
        "joints": ["L2", "L4"],
        "side": "L",
    },
}

LEG_NAMES = list(
    LEG_DEFS.keys()
)


# ============================================================
# 4. BASIC HELPERS
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
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. FLOATING BASE
# ============================================================

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
        "No free joint found."
    )


# ============================================================
# 6. ORIENTATION
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
        previous + delta
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

    quaternion = (
        data.xquat[
            body_id
        ].copy()
    )

    yaw = quat_to_yaw_wxyz(
        quaternion
    )

    rotation = (
        data.xmat[
            body_id
        ].reshape(
            3,
            3,
        )
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
# 7. BODY-FRAME MOTION
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
# 8. JOINT LIMITS
# ============================================================

def clip_joint_limits(q):

    q = np.array(
        q,
        dtype=float,
        copy=True,
    )

    limits = JOINT_LIMITS_RAD

    # dictionary form
    if isinstance(
        limits,
        dict,
    ):

        for name in JOINT_NAMES:

            if name not in limits:
                continue

            index = JOINT_INDEX[name]

            low, high = limits[name]

            q[index] = np.clip(
                q[index],
                low,
                high,
            )

        return q

    # array form
    limits = np.asarray(
        limits
    )

    if (
        limits.ndim == 2
        and limits.shape[0]
        == len(q)
        and limits.shape[1] >= 2
    ):

        return np.clip(
            q,
            limits[:, 0],
            limits[:, 1],
        )

    return q


# ============================================================
# 9. ACTUATOR MAP
# ============================================================

def build_actuator_map(model):

    result = {}

    for joint_name in JOINT_NAMES:

        # first try actuator name
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            joint_name,
        )

        if actuator_id >= 0:

            result[
                joint_name
            ] = actuator_id

            continue

        # fallback:
        # actuator transmission target
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

            if target_name == joint_name:

                result[
                    joint_name
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

    missing = [
        name
        for name in JOINT_NAMES
        if name not in result
    ]

    if missing:
        print(
            "[WARNING] Missing actuators:",
            missing,
        )

    return result


def apply_joint_targets(
    data,
    actuator_map,
    targets,
):

    for joint_name in JOINT_NAMES:

        actuator_id = (
            actuator_map.get(
                joint_name
            )
        )

        if actuator_id is None:
            continue

        index = JOINT_INDEX[
            joint_name
        ]

        data.ctrl[
            actuator_id
        ] = targets[
            index
        ]


# ============================================================
# 10. SERVO MODEL
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
# 11. SAME GAIT AS G2-E
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
        joint_name,
        value,
    ):

        if joint_name not in JOINT_INDEX:
            return

        q[
            JOINT_INDEX[
                joint_name
            ]
        ] += value

    # --------------------------------------------------------
    # proximal
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # distal
    # --------------------------------------------------------

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

    return clip_joint_limits(q)


# ============================================================
# 12. BODY-TREE / LEG ATTRIBUTION
# ============================================================

def body_ancestor_distance(
    model,
    child_body,
    ancestor_body,
):
    """
    Returns number of parent steps from child to ancestor.

    None = ancestor_body is not actually an ancestor.
    """

    body = int(
        child_body
    )

    distance = 0

    while body >= 0:

        if body == ancestor_body:
            return distance

        if body == 0:
            break

        body = int(
            model.body_parentid[
                body
            ]
        )

        distance += 1

    return None


def build_leg_root_bodies(model):

    roots = {}

    print()
    print(
        "=== LEG ROOT BODIES ==="
    )

    for leg_name, cfg in (
        LEG_DEFS.items()
    ):

        root_joint_name = (
            cfg["root_joint"]
        )

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            root_joint_name,
        )

        if joint_id < 0:

            print(
                f"[WARNING] Missing joint "
                f"{root_joint_name}"
            )

            continue

        body_id = int(
            model.jnt_bodyid[
                joint_id
            ]
        )

        body_name = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        roots[
            leg_name
        ] = body_id

        print(
            f"{leg_name:6s} "
            f"root_joint={root_joint_name:2s} "
            f"body={body_name} "
            f"(id={body_id})"
        )

    return roots


def build_geom_leg_map(
    model,
    leg_roots,
):
    """
    Assign every robot geom to the closest matching leg-root
    ancestor.

    This avoids relying on STL / geom naming conventions.
    """

    geom_to_leg = {}
    rows = []

    for geom_id in range(
        model.ngeom
    ):

        body_id = int(
            model.geom_bodyid[
                geom_id
            ]
        )

        geom_name = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        body_name = safe_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        candidates = []

        for leg_name, root_body in (
            leg_roots.items()
        ):

            distance = body_ancestor_distance(
                model,
                body_id,
                root_body,
            )

            if distance is not None:

                candidates.append(
                    (
                        distance,
                        leg_name,
                    )
                )

        leg = ""

        if candidates:

            # nearest ancestor wins
            candidates.sort(
                key=lambda x: x[0]
            )

            leg = candidates[0][1]

            geom_to_leg[
                geom_id
            ] = leg

        rows.append({
            "geom_id":
                geom_id,

            "geom_name":
                geom_name,

            "body_id":
                body_id,

            "body_name":
                body_name,

            "leg":
                leg,
        })

    return (
        geom_to_leg,
        rows,
    )


# ============================================================
# 13. GROUND DETECTION
# ============================================================

def is_ground_geom(
    model,
    geom_id,
):
    """
    Ground is usually attached directly to worldbody (body 0).

    Name matching is kept as fallback.
    """

    body_id = int(
        model.geom_bodyid[
            geom_id
        ]
    )

    if body_id == 0:
        return True

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


# ============================================================
# 14. PER-LEG CONTACT STATE
# ============================================================

def get_leg_contacts(
    model,
    data,
    geom_to_leg,
):
    """
    Returns for each leg:

    count
    contact binary
    normal-force sum

    Force is obtained with mj_contactForce().
    """

    result = {}

    for leg in LEG_NAMES:

        result[leg] = {
            "count": 0,
            "binary": 0,
            "normal_force": 0.0,
        }

    unassigned_ground_contacts = 0

    for contact_id in range(
        data.ncon
    ):

        contact = data.contact[
            contact_id
        ]

        g1 = int(
            contact.geom1
        )

        g2 = int(
            contact.geom2
        )

        ground1 = is_ground_geom(
            model,
            g1,
        )

        ground2 = is_ground_geom(
            model,
            g2,
        )

        # ignore robot-robot collisions
        if not (
            ground1
            or ground2
        ):
            continue

        # impossible / unhelpful world-world contact
        if ground1 and ground2:
            continue

        robot_geom = (
            g2
            if ground1
            else g1
        )

        leg = geom_to_leg.get(
            robot_geom
        )

        if leg is None:

            unassigned_ground_contacts += 1
            continue

        # Contact force in MuJoCo contact frame
        force6 = np.zeros(
            6,
            dtype=float,
        )

        mujoco.mj_contactForce(
            model,
            data,
            contact_id,
            force6,
        )

        # first component = contact-normal direction
        normal_force = abs(
            float(
                force6[0]
            )
        )

        result[leg]["count"] += 1
        result[leg]["binary"] = 1

        result[leg][
            "normal_force"
        ] += normal_force

    return (
        result,
        unassigned_ground_contacts,
    )


# ============================================================
# 15. CONTACT EVENTS
# ============================================================

def detect_contact_events(
    current,
    previous,
    step,
    time,
    cycle,
    phase,
    v_forward,
    v_lateral,
    yaw_rate_deg,
):
    events = []

    for leg in LEG_NAMES:

        now = bool(
            current[
                leg
            ]["binary"]
        )

        before = bool(
            previous.get(
                leg,
                False,
            )
        )

        if now == before:
            continue

        event_type = (
            "touchdown"
            if now
            else "liftoff"
        )

        events.append({
            "step":
                step,

            "time":
                time,

            "cycle":
                cycle,

            "phase":
                phase,

            "leg":
                leg,

            "event":
                event_type,

            "forward_velocity":
                v_forward,

            "lateral_velocity":
                v_lateral,

            "yaw_rate_deg_s":
                yaw_rate_deg,
        })

    return events


# ============================================================
# 16. PER-CYCLE ANALYSIS
# ============================================================

def analyse_cycles(log):

    cycles = []

    if not log:
        return cycles

    cycle_ids = sorted(
        set(
            row["cycle"]
            for row in log
        )
    )

    expected_steps = (
        GAIT_PERIOD
        / CONTROL_DT
    )

    for cycle_id in cycle_ids:

        rows = [
            row
            for row in log
            if row["cycle"]
            == cycle_id
        ]

        # reject incomplete cycle
        if (
            len(rows)
            < expected_steps * 0.8
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

        (
            delta_forward,
            delta_lateral,
        ) = world_delta_to_body(
            dx,
            dy,
            start["yaw"],
        )

        delta_yaw_deg = np.rad2deg(
            end["yaw"]
            - start["yaw"]
        )

        cycle_row = {
            "cycle":
                cycle_id,

            "start_time":
                start["time"],

            "end_time":
                end["time"],

            "delta_forward":
                delta_forward,

            "delta_lateral":
                delta_lateral,

            "delta_yaw_deg":
                delta_yaw_deg,
        }

        for leg in LEG_NAMES:

            binary = np.asarray([
                r[
                    f"{leg}_contact"
                ]
                for r in rows
            ])

            force = np.asarray([
                r[
                    f"{leg}_normal_force"
                ]
                for r in rows
            ])

            cycle_row[
                f"{leg}_duty"
            ] = float(
                np.mean(
                    binary
                )
            )

            cycle_row[
                f"{leg}_mean_force"
            ] = float(
                np.mean(
                    force
                )
            )

            if np.any(
                binary > 0
            ):

                cycle_row[
                    f"{leg}_force_when_contact"
                ] = float(
                    np.mean(
                        force[
                            binary > 0
                        ]
                    )
                )

            else:

                cycle_row[
                    f"{leg}_force_when_contact"
                ] = 0.0

        cycles.append(
            cycle_row
        )

    return cycles


# ============================================================
# 17. PHASE ANALYSIS
# ============================================================

def analyse_phase_bins(log):

    rows_out = []

    for bin_id in range(
        PHASE_BINS
    ):

        low = (
            bin_id
            / PHASE_BINS
        )

        high = (
            (bin_id + 1)
            / PHASE_BINS
        )

        selected = [
            r
            for r in log
            if (
                r["phase"] >= low
                and
                (
                    r["phase"] < high
                    or
                    (
                        bin_id
                        == PHASE_BINS - 1
                        and
                        r["phase"] <= 1.0
                    )
                )
            )
        ]

        if not selected:
            continue

        row = {
            "phase_bin":
                bin_id,

            "phase_start":
                low,

            "phase_end":
                high,

            "phase_center":
                (
                    low + high
                ) / 2.0,

            "samples":
                len(selected),

            "mean_forward_velocity":
                float(
                    np.mean([
                        r[
                            "body_forward_velocity"
                        ]
                        for r in selected
                    ])
                ),

            "mean_lateral_velocity":
                float(
                    np.mean([
                        r[
                            "body_lateral_velocity"
                        ]
                        for r in selected
                    ])
                ),

            "mean_yaw_rate_deg_s":
                float(
                    np.mean([
                        r[
                            "yaw_rate_deg_s"
                        ]
                        for r in selected
                    ])
                ),
        }

        for leg in LEG_NAMES:

            contact = np.asarray([
                r[
                    f"{leg}_contact"
                ]
                for r in selected
            ])

            force = np.asarray([
                r[
                    f"{leg}_normal_force"
                ]
                for r in selected
            ])

            row[
                f"{leg}_contact_probability"
            ] = float(
                np.mean(contact)
            )

            row[
                f"{leg}_mean_normal_force"
            ] = float(
                np.mean(force)
            )

        rows_out.append(
            row
        )

    return rows_out


# ============================================================
# 18. CONTACT-CONDITIONED ANALYSIS
# ============================================================

def print_contact_conditioned_summary(
    log,
):

    print()
    print(
        "--- CONTACT-CONDITIONED MOTION ---"
    )

    for leg in LEG_NAMES:

        active = [
            r
            for r in log
            if r[
                f"{leg}_contact"
            ] > 0
        ]

        if not active:

            print(
                f"{leg:6s}: "
                f"NO GROUND CONTACT DETECTED"
            )

            continue

        vf = np.asarray([
            r[
                "body_forward_velocity"
            ]
            for r in active
        ])

        vl = np.asarray([
            r[
                "body_lateral_velocity"
            ]
            for r in active
        ])

        yaw_rate = np.asarray([
            r[
                "yaw_rate_deg_s"
            ]
            for r in active
        ])

        force = np.asarray([
            r[
                f"{leg}_normal_force"
            ]
            for r in active
        ])

        print(
            f"{leg:6s} "
            f"duty={len(active)/len(log):.3f} "
            f"| vf={np.mean(vf):+.4f} m/s "
            f"| vl={np.mean(vl):+.4f} m/s "
            f"| yaw_rate={np.mean(yaw_rate):+.2f} deg/s "
            f"| Fnormal={np.mean(force):.3f}"
        )


# ============================================================
# 19. MAIN SUMMARY
# ============================================================

def print_summary(
    log,
    cycles,
    events,
):

    print()
    print("=" * 74)
    print(
        "05 PER-LEG CONTACT & PHASE DIAGNOSTIC"
    )
    print("=" * 74)

    print(
        f"Control steps             : "
        f"{len(log)}"
    )

    print(
        f"Complete gait cycles      : "
        f"{len(cycles)}"
    )

    print(
        f"Contact events            : "
        f"{len(events)}"
    )

    print()
    print(
        "--- PER-LEG CONTACT DUTY ---"
    )

    for leg in LEG_NAMES:

        contact = np.asarray([
            r[
                f"{leg}_contact"
            ]
            for r in log
        ])

        force = np.asarray([
            r[
                f"{leg}_normal_force"
            ]
            for r in log
        ])

        duty = float(
            np.mean(contact)
        )

        if np.any(
            contact > 0
        ):

            force_active = float(
                np.mean(
                    force[
                        contact > 0
                    ]
                )
            )

        else:
            force_active = 0.0

        print(
            f"{leg:6s} "
            f"duty={duty:.4f} "
            f"| mean force when contact="
            f"{force_active:.3f}"
        )

    # --------------------------------------------------------
    # right / left
    # --------------------------------------------------------

    right_binary = np.asarray([
        max(
            r["R1-R4_contact"],
            r["R2-R3_contact"],
        )
        for r in log
    ])

    left_binary = np.asarray([
        max(
            r["L1-L3_contact"],
            r["L2-L4_contact"],
        )
        for r in log
    ])

    right_duty = float(
        np.mean(
            right_binary
        )
    )

    left_duty = float(
        np.mean(
            left_binary
        )
    )

    print()
    print(
        "--- SIDE SUMMARY ---"
    )

    print(
        f"Right-side duty           : "
        f"{right_duty:.4f}"
    )

    print(
        f"Left-side duty            : "
        f"{left_duty:.4f}"
    )

    print(
        f"R-L duty difference       : "
        f"{right_duty-left_duty:+.4f}"
    )

    if cycles:

        delta_forward = np.asarray([
            c["delta_forward"]
            for c in cycles
        ])

        delta_lateral = np.asarray([
            c["delta_lateral"]
            for c in cycles
        ])

        delta_yaw = np.asarray([
            c["delta_yaw_deg"]
            for c in cycles
        ])

        print()
        print(
            "--- PER-CYCLE MOTION ---"
        )

        print(
            f"Mean Δforward / cycle     : "
            f"{np.mean(delta_forward):+.5f} m"
        )

        print(
            f"Mean Δlateral / cycle     : "
            f"{np.mean(delta_lateral):+.5f} m"
        )

        print(
            f"Mean |Δlateral| / cycle   : "
            f"{np.mean(np.abs(delta_lateral)):.5f} m"
        )

        print(
            f"Mean Δyaw / cycle         : "
            f"{np.mean(delta_yaw):+.3f} deg"
        )

        print(
            f"Mean |Δyaw| / cycle       : "
            f"{np.mean(np.abs(delta_yaw)):.3f} deg"
        )

        print(
            f"Cycles with +lateral      : "
            f"{np.mean(delta_lateral > 0):.3f}"
        )

        print(
            f"Cycles with +yaw          : "
            f"{np.mean(delta_yaw > 0):.3f}"
        )

    print_contact_conditioned_summary(
        log
    )

    print()
    print("=" * 74)


# ============================================================
# 20. PLOTTING
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


def plot_contact_raster(log):

    time = np.asarray([
        r["time"]
        for r in log
    ])

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    offsets = {
        "R1-R4": 0,
        "R2-R3": 2,
        "L1-L3": 4,
        "L2-L4": 6,
    }

    for leg in LEG_NAMES:

        contact = np.asarray([
            r[
                f"{leg}_contact"
            ]
            for r in log
        ])

        ax.step(
            time,
            contact
            + offsets[leg],
            where="post",
            label=leg,
        )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_yticks([
        0.5,
        2.5,
        4.5,
        6.5,
    ])

    ax.set_yticklabels(
        LEG_NAMES
    )

    ax.set_title(
        "05 Per-Leg Ground Contact Timeline"
    )

    ax.grid(True)

    save_figure(
        fig,
        "05_contact_raster.png",
    )


def plot_phase_contact_probability(
    phase_rows,
):

    phase = np.asarray([
        r["phase_center"]
        for r in phase_rows
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for leg in LEG_NAMES:

        probability = np.asarray([
            r[
                f"{leg}_contact_probability"
            ]
            for r in phase_rows
        ])

        ax.plot(
            phase,
            probability,
            marker="o",
            label=leg,
        )

    ax.set_xlabel(
        "Normalized gait phase"
    )

    ax.set_ylabel(
        "Contact probability"
    )

    ax.set_ylim(
        -0.05,
        1.05,
    )

    ax.set_title(
        "05 Contact Probability by Gait Phase"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "05_phase_contact_probability.png",
    )


def plot_phase_force(
    phase_rows,
):

    phase = np.asarray([
        r["phase_center"]
        for r in phase_rows
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for leg in LEG_NAMES:

        force = np.asarray([
            r[
                f"{leg}_mean_normal_force"
            ]
            for r in phase_rows
        ])

        ax.plot(
            phase,
            force,
            marker="o",
            label=leg,
        )

    ax.set_xlabel(
        "Normalized gait phase"
    )

    ax.set_ylabel(
        "Mean normal contact force"
    )

    ax.set_title(
        "05 Normal Contact Force by Gait Phase"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "05_phase_normal_force.png",
    )


def plot_phase_motion(
    phase_rows,
):

    phase = np.asarray([
        r["phase_center"]
        for r in phase_rows
    ])

    forward = np.asarray([
        r[
            "mean_forward_velocity"
        ]
        for r in phase_rows
    ])

    lateral = np.asarray([
        r[
            "mean_lateral_velocity"
        ]
        for r in phase_rows
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        phase,
        forward,
        marker="o",
        label="Forward velocity",
    )

    ax.plot(
        phase,
        lateral,
        marker="o",
        label="Lateral velocity",
    )

    ax.axhline(
        0.0
    )

    ax.set_xlabel(
        "Normalized gait phase"
    )

    ax.set_ylabel(
        "Velocity [m/s]"
    )

    ax.set_title(
        "05 Body Motion by Gait Phase"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "05_phase_motion.png",
    )


def plot_phase_yaw(
    phase_rows,
):

    phase = np.asarray([
        r["phase_center"]
        for r in phase_rows
    ])

    yaw_rate = np.asarray([
        r[
            "mean_yaw_rate_deg_s"
        ]
        for r in phase_rows
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        phase,
        yaw_rate,
        marker="o",
    )

    ax.axhline(
        0.0
    )

    ax.set_xlabel(
        "Normalized gait phase"
    )

    ax.set_ylabel(
        "Yaw rate [deg/s]"
    )

    ax.set_title(
        "05 Yaw Rate by Gait Phase"
    )

    ax.grid(True)

    save_figure(
        fig,
        "05_phase_yaw_rate.png",
    )


def plot_touchdown_phases(
    events,
):

    touchdowns = [
        e
        for e in events
        if e["event"]
        == "touchdown"
    ]

    if not touchdowns:
        return

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    y_map = {
        "R1-R4": 0,
        "R2-R3": 1,
        "L1-L3": 2,
        "L2-L4": 3,
    }

    for leg in LEG_NAMES:

        phases = [
            e["phase"]
            for e in touchdowns
            if e["leg"] == leg
        ]

        ys = [
            y_map[leg]
            for _ in phases
        ]

        ax.scatter(
            phases,
            ys,
            label=leg,
        )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_yticks(
        list(
            y_map.values()
        )
    )

    ax.set_yticklabels(
        list(
            y_map.keys()
        )
    )

    ax.set_xlabel(
        "Touchdown gait phase"
    )

    ax.set_title(
        "05 Touchdown Phase Distribution"
    )

    ax.grid(True)

    save_figure(
        fig,
        "05_touchdown_phase.png",
    )


def plot_cycle_motion(
    cycles,
):

    if not cycles:
        return

    cycle = np.asarray([
        c["cycle"]
        for c in cycles
    ])

    forward = np.asarray([
        c["delta_forward"]
        for c in cycles
    ])

    lateral = np.asarray([
        c["delta_lateral"]
        for c in cycles
    ])

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        cycle,
        forward,
        marker="o",
        label="Δforward",
    )

    ax.plot(
        cycle,
        lateral,
        marker="o",
        label="Δlateral",
    )

    ax.axhline(
        0.0
    )

    ax.set_xlabel(
        "Gait cycle"
    )

    ax.set_ylabel(
        "Body displacement [m]"
    )

    ax.set_title(
        "05 Per-Cycle Body Displacement"
    )

    ax.grid(True)
    ax.legend()

    save_figure(
        fig,
        "05_cycle_motion.png",
    )


# ============================================================
# 21. MAIN
# ============================================================

def main():

    print()
    print(
        "=== 05 PER-LEG CONTACT & PHASE DIAGNOSTIC ==="
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
        f"Gait frequency   : "
        f"{GAIT_FREQUENCY:.3f} Hz"
    )

    print(
        f"Gait period      : "
        f"{GAIT_PERIOD:.4f} s"
    )

    print(
        f"Phase bins       : "
        f"{PHASE_BINS}"
    )

    print(
        f"Simulation       : "
        f"{SIM_DURATION:.1f} s"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "[1] Loading sesame_ml model"
    )

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

    # ========================================================
    # BASE
    # ========================================================

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
        f"Base body        : "
        f"{base_name}"
    )

    # ========================================================
    # ACTUATORS
    # ========================================================

    actuator_map = (
        build_actuator_map(
            model
        )
    )

    # ========================================================
    # LEG / GEOM MAP
    # ========================================================

    leg_roots = (
        build_leg_root_bodies(
            model
        )
    )

    (
        geom_to_leg,
        geom_map_rows,
    ) = build_geom_leg_map(
        model,
        leg_roots,
    )

    save_csv(
        GEOM_MAP_CSV,
        geom_map_rows,
    )

    mapped_geoms = len(
        geom_to_leg
    )

    print()
    print(
        f"Mapped leg geoms : "
        f"{mapped_geoms}"
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

    previous_yaw = float(
        start_yaw
    )

    print(
        f"Start XYZ : "
        f"{previous_position}"
    )

    print(
        f"Start yaw : "
        f"{np.rad2deg(start_yaw):.2f} deg"
    )

    print(
        f"Upright   : "
        f"{start_upright:.4f}"
    )

    # ========================================================
    # FIXED STEP COUNT
    # ========================================================

    num_steps = int(
        round(
            SIM_DURATION
            / CONTROL_DT
        )
    )

    print()
    print(
        f"[3] Running "
        f"{num_steps} control steps"
    )

    log = []
    events = []

    previous_contact_state = {
        leg: False
        for leg in LEG_NAMES
    }

    last_print = -999.0

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

            # =================================================
            # COMMAND
            # =================================================

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

            # =================================================
            # PHYSICS
            # =================================================

            for _ in range(
                SUBSTEPS
            ):

                mujoco.mj_step(
                    model,
                    data,
                )

            # =================================================
            # BASE STATE
            # =================================================

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

            yaw_rate_deg = float(
                np.rad2deg(
                    yaw_rate
                )
            )

            # =================================================
            # PER-LEG CONTACT
            # =================================================

            (
                leg_contact,
                unassigned_contacts,
            ) = get_leg_contacts(
                model,
                data,
                geom_to_leg,
            )

            # =================================================
            # CONTACT EVENTS
            # =================================================

            new_events = (
                detect_contact_events(
                    leg_contact,
                    previous_contact_state,
                    step,
                    sim_time,
                    cycle_id,
                    phase,
                    v_forward,
                    v_lateral,
                    yaw_rate_deg,
                )
            )

            events.extend(
                new_events
            )

            # =================================================
            # STEP LOG
            # =================================================

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

                "yaw_deg":
                    float(
                        np.rad2deg(
                            yaw
                        )
                    ),

                "yaw_rate_deg_s":
                    yaw_rate_deg,

                "upright":
                    float(
                        upright
                    ),

                "unassigned_ground_contacts":
                    int(
                        unassigned_contacts
                    ),
            }

            for leg in LEG_NAMES:

                row[
                    f"{leg}_contact"
                ] = int(
                    leg_contact[
                        leg
                    ]["binary"]
                )

                row[
                    f"{leg}_contact_count"
                ] = int(
                    leg_contact[
                        leg
                    ]["count"]
                )

                row[
                    f"{leg}_normal_force"
                ] = float(
                    leg_contact[
                        leg
                    ]["normal_force"]
                )

            log.append(
                row
            )

            # =================================================
            # TERMINAL
            # =================================================

            if (
                sim_time
                - last_print
                >= PRINT_INTERVAL
            ):

                contact_text = " ".join([
                    f"{leg}="
                    f"{leg_contact[leg]['binary']}"
                    for leg in LEG_NAMES
                ])

                print(
                    f"t={sim_time:5.1f}s "
                    f"cycle={cycle_id:02d} "
                    f"phase={phase:.2f} "
                    f"| vf={v_forward:+.4f} "
                    f"vl={v_lateral:+.4f} "
                    f"| yawRate={yaw_rate_deg:+7.2f} "
                    f"| {contact_text}"
                )

                last_print = sim_time

            # =================================================
            # UPDATE
            # =================================================

            previous_position = (
                position.copy()
            )

            previous_yaw = yaw

            previous_contact_state = {
                leg: bool(
                    leg_contact[
                        leg
                    ]["binary"]
                )
                for leg in LEG_NAMES
            }

            viewer.sync()

    finally:

        viewer.close()

    # ========================================================
    # ANALYSIS
    # ========================================================

    print()
    print(
        "[4] Analysing cycles and phase"
    )

    cycles = analyse_cycles(
        log
    )

    phase_rows = analyse_phase_bins(
        log
    )

    # ========================================================
    # EXPORT
    # ========================================================

    save_csv(
        STEP_CSV,
        log,
    )

    save_csv(
        CYCLE_CSV,
        cycles,
    )

    save_csv(
        PHASE_CSV,
        phase_rows,
    )

    save_csv(
        EVENT_CSV,
        events,
    )

    # ========================================================
    # PLOTS
    # ========================================================

    if log:

        plot_contact_raster(
            log
        )

    if phase_rows:

        plot_phase_contact_probability(
            phase_rows
        )

        plot_phase_force(
            phase_rows
        )

        plot_phase_motion(
            phase_rows
        )

        plot_phase_yaw(
            phase_rows
        )

    if events:

        plot_touchdown_phases(
            events
        )

    if cycles:

        plot_cycle_motion(
            cycles
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        log,
        cycles,
        events,
    )

    # ========================================================
    # VERIFICATION
    # ========================================================

    print()
    print(
        "=== 05 VERIFICATION ==="
    )

    print(
        f"Control steps       : "
        f"{len(log)} / {num_steps}"
    )

    print(
        f"Complete cycles     : "
        f"{len(cycles)}"
    )

    print(
        f"Phase bins          : "
        f"{len(phase_rows)} / "
        f"{PHASE_BINS}"
    )

    print(
        f"Contact events      : "
        f"{len(events)}"
    )

    print(
        f"Mapped robot geoms  : "
        f"{mapped_geoms}"
    )

    if log:

        unassigned_total = sum(
            r[
                "unassigned_ground_contacts"
            ]
            for r in log
        )

    else:

        unassigned_total = 0

    print(
        f"Unassigned contacts : "
        f"{unassigned_total}"
    )

    print(
        f"Step CSV            : "
        f"{'PASS' if STEP_CSV.exists() else 'FAIL'}"
    )

    print(
        f"Cycle CSV           : "
        f"{'PASS' if CYCLE_CSV.exists() else 'FAIL'}"
    )

    print(
        f"Phase CSV           : "
        f"{'PASS' if PHASE_CSV.exists() else 'FAIL'}"
    )

    print(
        f"Event CSV           : "
        f"{'PASS' if EVENT_CSV.exists() else 'FAIL'}"
    )

    print(
        f"Geom map CSV        : "
        f"{'PASS' if GEOM_MAP_CSV.exists() else 'FAIL'}"
    )

    print()
    print(
        f"Output folder:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()