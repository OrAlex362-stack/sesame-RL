import time
import numpy as np
import mujoco
from mujoco import viewer

from sesame_ml.model import load_model, make_data
from sesame_ml.constants import (
    JOINT_NAMES,
    JOINT_INDEX,
    STAND_ANGLES_DEG,
    STAND_ANGLES_RAD,
    JOINT_LIMITS_RAD,
)


# ============================================================
# G2-A — Firmware gait viewer
# ============================================================

FRAME_DURATION = 0.1000   # Original firmware frameDelay = 100 ms
WALK_CYCLES = 20         # Original firmware default
START_STAND_S = 5.0
END_STAND_S = 10.0


# ------------------------------------------------------------
# Original Sesame firmware forward gait
#
# Directly based on:
# firmware/movement-sequences.h -> runWalkPose()
#
# IMPORTANT:
# These updates are STATEFUL.
# A joint not mentioned in a frame keeps its previous target.
# ------------------------------------------------------------

FORWARD_INITIAL = {
    "R3": 135,
    "L3": 45,
    "R2": 100,
    "L1": 25,
}

FORWARD_CYCLE = (
    {
        "R3": 135,
        "L3": 0,
    },
    {
        "L4": 135,
        "L2": 90,
        "R4": 0,
        "R1": 180,
    },
    {
        "R2": 45,
        "L1": 90,
    },
    {
        "R4": 45,
        "L4": 180,
    },
    {
        "R3": 180,
        "L3": 45,
        "R2": 90,
        "L1": 0,
    },
    {
        "L2": 135,
        "R1": 90,
    },
)


# ============================================================
# Load MuJoCo Sesame
# ============================================================

model = load_model()
data = make_data(model, settle=True)


# ------------------------------------------------------------
# Verify actuator mapping before walking
# ------------------------------------------------------------

expected_actuators = [
    f"{name}_servo"
    for name in JOINT_NAMES
]

actual_actuators = [
    mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i,
    )
    for i in range(model.nu)
]

if actual_actuators != expected_actuators:
    raise RuntimeError(
        "Actuator order mismatch!\n"
        f"Expected: {expected_actuators}\n"
        f"Actual:   {actual_actuators}"
    )


print("\n=== G2-A FIRMWARE GAIT VIEW ===\n")

print("Actuator order verified:")

for i, name in enumerate(actual_actuators):
    print(f"{i}: {name}")

print(f"\nPhysics timestep : {model.opt.timestep:.4f} s")
print(f"Firmware frame  : {FRAME_DURATION:.3f} s")
print(f"Walking cycles  : {WALK_CYCLES}")


# ============================================================
# Helpers
# ============================================================

def clip_targets(target_rad):
    return np.clip(
        target_rad,
        JOINT_LIMITS_RAD[:, 0],
        JOINT_LIMITS_RAD[:, 1],
    )


def simulate_target(v, target_rad, duration):
    """
    Hold one servo target while MuJoCo physics keeps running.
    """

    target_rad = clip_targets(target_rad)

    steps = round(
        duration / model.opt.timestep
    )

    for _ in range(steps):

        if not v.is_running():
            return False

        start = time.perf_counter()

        data.ctrl[:] = target_rad

        mujoco.mj_step(
            model,
            data,
        )

        v.sync()

        elapsed = time.perf_counter() - start

        remaining = (
            model.opt.timestep
            - elapsed
        )

        if remaining > 0:
            time.sleep(remaining)

    return True


def apply_updates(current_deg, updates):
    """
    Reproduce firmware setServoAngle() behavior.

    Only joints present in `updates` change.
    All other joints keep their previous target.
    """

    new_pose = current_deg.copy()

    for joint_name, angle_deg in updates.items():

        index = JOINT_INDEX[joint_name]

        new_pose[index] = angle_deg

    return new_pose


def print_frame(label, pose_deg):
    values = " ".join(
        f"{name}={pose_deg[i]:6.1f}"
        for i, name in enumerate(JOINT_NAMES)
    )

    print(
        f"{label:<18} | {values}"
    )


# ============================================================
# Viewer
# ============================================================

with viewer.launch_passive(
    model,
    data,
) as v:

    # --------------------------------------------------------
    # 1. Initial stand
    # --------------------------------------------------------

    print("\n[1] Initial stand")

    current_deg = (
        STAND_ANGLES_DEG.copy()
    )

    print_frame(
        "STAND",
        current_deg,
    )

    if not simulate_target(
        v,
        STAND_ANGLES_RAD,
        START_STAND_S,
    ):
        raise SystemExit


    # Record initial robot position
    start_position = (
        data.qpos[0:3].copy()
    )


    # --------------------------------------------------------
    # 2. Firmware initial step
    # --------------------------------------------------------

    print("\n[2] Firmware initial step")

    current_deg = apply_updates(
        current_deg,
        FORWARD_INITIAL,
    )

    print_frame(
        "INITIAL",
        current_deg,
    )

    if not simulate_target(
        v,
        np.deg2rad(current_deg),
        FRAME_DURATION,
    ):
        raise SystemExit


    # --------------------------------------------------------
    # 3. Firmware walking cycles
    # --------------------------------------------------------

    print("\n[3] Walking")

    for cycle in range(WALK_CYCLES):

        print(
            f"\n--- Cycle "
            f"{cycle + 1}/{WALK_CYCLES} ---"
        )

        for frame_index, updates in enumerate(
            FORWARD_CYCLE,
            start=1,
        ):

            current_deg = apply_updates(
                current_deg,
                updates,
            )

            print_frame(
                f"Frame {frame_index}",
                current_deg,
            )

            running = simulate_target(
                v,
                np.deg2rad(current_deg),
                FRAME_DURATION,
            )

            if not running:
                raise SystemExit


    # --------------------------------------------------------
    # 4. Return to stand
    # --------------------------------------------------------

    print("\n[4] Return to stand")

    current_deg = (
        STAND_ANGLES_DEG.copy()
    )

    print_frame(
        "STAND",
        current_deg,
    )

    simulate_target(
        v,
        STAND_ANGLES_RAD,
        END_STAND_S,
    )


    # --------------------------------------------------------
    # 5. Rough displacement preview
    #
    # NOT yet the formal G2 evaluation.
    # --------------------------------------------------------

    end_position = (
        data.qpos[0:3].copy()
    )

    displacement = (
        end_position
        - start_position
    )

    print("\n=== PREVIEW ===")

    print(
        f"Start XYZ : "
        f"{start_position}"
    )

    print(
        f"End XYZ   : "
        f"{end_position}"
    )

    print(
        f"Delta XYZ : "
        f"{displacement}"
    )

    print(
        f"Forward Δx: "
        f"{displacement[0]:.4f} m"
    )

    print(
        f"Lateral Δy: "
        f"{displacement[1]:.4f} m"
    )

    print("\n=== G2-A COMPLETE ===")
    print(
        "Close the MuJoCo viewer "
        "to exit."
    )


    # --------------------------------------------------------
    # Keep robot standing
    # --------------------------------------------------------

    while v.is_running():

        simulate_target(
            v,
            STAND_ANGLES_RAD,
            0.02,
        )