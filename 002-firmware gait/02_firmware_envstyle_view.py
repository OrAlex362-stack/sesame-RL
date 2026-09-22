import time
import math
import numpy as np
import mujoco
from mujoco import viewer

from sesame_ml.model import load_model, make_data
from sesame_ml.constants import (
    JOINT_INDEX,
    JOINT_NAMES,
    STAND_ANGLES_DEG,
    STAND_ANGLES_RAD,
    JOINT_LIMITS_RAD,
)


# ============================================================
# G2-C CONFIG
# ============================================================

CONTROL_DT = 0.020          # 50 Hz
ACTION_SCALE_RAD = 0.58

SERVO_TIME_CONSTANT = 0.045
SERVO_MAX_SPEED_RAD_S = np.deg2rad(600.0)

FRAME_DURATION = 0.100

SIM_DURATION = 20.0
START_STAND_DURATION = 2.0


# ============================================================
# ORIGINAL FIRMWARE FORWARD GAIT
# ============================================================

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
# MODEL
# ============================================================

model = load_model()
data = make_data(model, settle=True)

physics_dt = model.opt.timestep

substeps = round(
    CONTROL_DT / physics_dt
)


print("\n=== G2-C ENV-STYLE FIRMWARE BASELINE ===")

print(f"Physics dt       : {physics_dt:.4f} s")
print(f"Control dt       : {CONTROL_DT:.4f} s")
print(f"Substeps         : {substeps}")
print(f"Servo tau        : {SERVO_TIME_CONSTANT:.3f} s")
print(
    f"Servo max speed  : "
    f"{np.rad2deg(SERVO_MAX_SPEED_RAD_S):.1f} deg/s"
)
print(f"Action scale     : ±{ACTION_SCALE_RAD:.3f} rad")
print(f"Simulation       : {SIM_DURATION:.1f} s")


# ============================================================
# FIRMWARE SEQUENCE
# ============================================================

def apply_updates(pose, updates):

    result = pose.copy()

    for joint_name, value in updates.items():

        result[
            JOINT_INDEX[joint_name]
        ] = value

    return result


def firmware_pose_at_frame(frame_index):

    """
    Reproduce FirmwareSequencePolicy._pose_at_frame()
    """

    pose = STAND_ANGLES_DEG.copy()

    pose = apply_updates(
        pose,
        FORWARD_INITIAL,
    )

    for i in range(frame_index):

        updates = FORWARD_CYCLE[
            i % len(FORWARD_CYCLE)
        ]

        pose = apply_updates(
            pose,
            updates,
        )

    return pose


def firmware_target(elapsed):

    """
    Interpolated firmware target.

    Same basic concept as FirmwareSequencePolicy.
    """

    frame_position = (
        elapsed / FRAME_DURATION
    )

    lower_index = math.floor(
        frame_position
    )

    fraction = (
        frame_position
        - lower_index
    )

    lower = firmware_pose_at_frame(
        lower_index
    )

    upper = firmware_pose_at_frame(
        lower_index + 1
    )

    target_deg = (
        lower
        + fraction * (upper - lower)
    )

    return np.deg2rad(target_deg)


# ============================================================
# ENV ACTION CONVERSION
# ============================================================

def firmware_to_gym_target(
    firmware_target_rad
):

    """
    FirmwareSequencePolicy does NOT send the complete
    firmware angle directly into SesameEnv.

    It converts target to residual action:

        residual =
        (firmware - stand) / 0.58

    then clips it to [-1, 1].

    SesameEnv then converts it back:

        stand + action * 0.58
    """

    residual = (
        firmware_target_rad
        - STAND_ANGLES_RAD
    ) / ACTION_SCALE_RAD

    residual = np.clip(
        residual,
        -1.0,
        1.0,
    )

    desired_target = (
        STAND_ANGLES_RAD
        + residual * ACTION_SCALE_RAD
    )

    desired_target = np.clip(
        desired_target,
        JOINT_LIMITS_RAD[:, 0],
        JOINT_LIMITS_RAD[:, 1],
    )

    return desired_target, residual


# ============================================================
# SERVO MODEL
# ============================================================

commanded_target = (
    STAND_ANGLES_RAD.copy()
)


def servo_filter(
    desired_target
):

    global commanded_target

    previous = commanded_target

    # First-order servo response
    alpha = (
        1.0
        - np.exp(
            -CONTROL_DT
            / SERVO_TIME_CONSTANT
        )
    )

    lagged = (
        previous
        + alpha
        * (
            desired_target
            - previous
        )
    )

    # MG90S speed limit
    maximum_delta = (
        SERVO_MAX_SPEED_RAD_S
        * CONTROL_DT
    )

    delta = np.clip(
        lagged - previous,
        -maximum_delta,
        maximum_delta,
    )

    commanded_target = (
        previous + delta
    )

    commanded_target = np.clip(
        commanded_target,
        JOINT_LIMITS_RAD[:, 0],
        JOINT_LIMITS_RAD[:, 1],
    )

    return commanded_target.copy()


# ============================================================
# PHYSICS
# ============================================================

def physics_step(
    v,
    control_target,
):

    start = time.perf_counter()

    data.ctrl[:] = control_target

    for _ in range(substeps):

        mujoco.mj_step(
            model,
            data,
        )

    v.sync()

    elapsed = (
        time.perf_counter()
        - start
    )

    remaining = (
        CONTROL_DT
        - elapsed
    )

    if remaining > 0:

        time.sleep(
            remaining
        )


# ============================================================
# YAW
# ============================================================

def quaternion_to_yaw(q):

    # MuJoCo:
    # w, x, y, z

    w, x, y, z = q

    siny = (
        2.0
        * (
            w * z
            + x * y
        )
    )

    cosy = (
        1.0
        - 2.0
        * (
            y * y
            + z * z
        )
    )

    return np.arctan2(
        siny,
        cosy,
    )


# ============================================================
# VIEWER
# ============================================================

with viewer.launch_passive(
    model,
    data,
) as v:

    print("\n[1] Initial stand")

    stand_steps = round(
        START_STAND_DURATION
        / CONTROL_DT
    )

    for _ in range(
        stand_steps
    ):

        physics_step(
            v,
            STAND_ANGLES_RAD,
        )

    start_xyz = (
        data.qpos[:3].copy()
    )

    start_yaw = quaternion_to_yaw(
        data.qpos[3:7]
    )

    previous_xy = (
        data.qpos[:2].copy()
    )

    path_distance = 0.0

    print(
        f"Start XYZ : "
        f"{start_xyz}"
    )

    print(
        f"Start yaw : "
        f"{np.rad2deg(start_yaw):.2f} deg"
    )


    # --------------------------------------------------------
    # Firmware policy simulation
    # --------------------------------------------------------

    print("\n[2] FirmwareSequencePolicy-style control")

    steps = round(
        SIM_DURATION
        / CONTROL_DT
    )

    for step in range(
        steps
    ):

        if not v.is_running():
            break

        elapsed = (
            step * CONTROL_DT
        )

        raw_firmware_target = (
            firmware_target(
                elapsed
            )
        )

        desired_target, residual = (
            firmware_to_gym_target(
                raw_firmware_target
            )
        )

        target = servo_filter(
            desired_target
        )

        physics_step(
            v,
            target,
        )

        current_xy = (
            data.qpos[:2].copy()
        )

        path_distance += (
            np.linalg.norm(
                current_xy
                - previous_xy
            )
        )

        previous_xy = (
            current_xy
        )

        # Print once per second
        if (
            step
            % round(
                1.0 / CONTROL_DT
            )
            == 0
        ):

            print(
                f"t={elapsed:5.1f}s "
                f"x={data.qpos[0]: .4f} "
                f"y={data.qpos[1]: .4f} "
                f"z={data.qpos[2]: .4f}"
            )


    # ========================================================
    # RESULT
    # ========================================================

    end_xyz = (
        data.qpos[:3].copy()
    )

    end_yaw = quaternion_to_yaw(
        data.qpos[3:7]
    )

    delta = (
        end_xyz
        - start_xyz
    )

    yaw_delta = np.arctan2(
        np.sin(
            end_yaw - start_yaw
        ),
        np.cos(
            end_yaw - start_yaw
        ),
    )

    print("\n================================")
    print("G2-C RESULT")
    print("================================")

    print(
        f"Start XYZ    : "
        f"{start_xyz}"
    )

    print(
        f"End XYZ      : "
        f"{end_xyz}"
    )

    print(
        f"Delta XYZ    : "
        f"{delta}"
    )

    print()

    print(
        f"Forward Δx   : "
        f"{delta[0]:.4f} m"
    )

    print(
        f"Lateral Δy   : "
        f"{delta[1]:.4f} m"
    )

    print(
        f"Path distance: "
        f"{path_distance:.4f} m"
    )

    print()

    print(
        f"Start yaw    : "
        f"{np.rad2deg(start_yaw):.2f} deg"
    )

    print(
        f"End yaw      : "
        f"{np.rad2deg(end_yaw):.2f} deg"
    )

    print(
        f"Yaw change   : "
        f"{np.rad2deg(yaw_delta):.2f} deg"
    )

    print("\n=== G2-C COMPLETE ===")

    while v.is_running():

        physics_step(
            v,
            commanded_target,
        )