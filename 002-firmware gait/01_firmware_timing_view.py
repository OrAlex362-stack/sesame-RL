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


FRAME_DELAY = 0.100
MOTOR_CURRENT_DELAY = 0.020
WALK_CYCLES = 20


FORWARD_INITIAL = (
    ("R3", 135),
    ("L3", 45),
    ("R2", 100),
    ("L1", 25),
)

FORWARD_CYCLE = (
    (
        ("R3", 135),
        ("L3", 0),
    ),
    (
        ("L4", 135),
        ("L2", 90),
        ("R4", 0),
        ("R1", 180),
    ),
    (
        ("R2", 45),
        ("L1", 90),
    ),
    (
        ("R4", 45),
        ("L4", 180),
    ),
    (
        ("R3", 180),
        ("L3", 45),
        ("R2", 90),
        ("L1", 0),
    ),
    (
        ("L2", 135),
        ("R1", 90),
    ),
)


model = load_model()
data = make_data(model, settle=True)

current_deg = STAND_ANGLES_DEG.copy()


def simulate(v, seconds):
    steps = round(seconds / model.opt.timestep)

    for _ in range(steps):
        if not v.is_running():
            return False

        start = time.perf_counter()

        target = np.deg2rad(current_deg)

        target = np.clip(
            target,
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1],
        )

        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        v.sync()

        remaining = model.opt.timestep - (
            time.perf_counter() - start
        )

        if remaining > 0:
            time.sleep(remaining)

    return True


def firmware_set_servo(v, joint_name, angle_deg):
    global current_deg

    idx = JOINT_INDEX[joint_name]

    current_deg[idx] = angle_deg

    print(
        f"    setServoAngle({joint_name}, {angle_deg})"
    )

    return simulate(
        v,
        MOTOR_CURRENT_DELAY,
    )


def execute_group(v, commands, label):
    print(label)

    for joint_name, angle_deg in commands:

        if not firmware_set_servo(
            v,
            joint_name,
            angle_deg,
        ):
            return False

    # equivalent to pressingCheck(..., frameDelay)
    return simulate(
        v,
        FRAME_DELAY,
    )


with viewer.launch_passive(model, data) as v:

    print("\n=== G2-B TIMING-FAITHFUL FIRMWARE GAIT ===")

    print("\n[1] Stand settle")

    simulate(v, 2.0)

    start_xyz = data.qpos[:3].copy()
    start_time = data.time

    print(
        f"Start XYZ = {start_xyz}"
    )

    print("\n[2] Initial firmware step")

    execute_group(
        v,
        FORWARD_INITIAL,
        "INITIAL",
    )

    print("\n[3] Walking")

    for cycle in range(WALK_CYCLES):

        print(
            f"\n--- Cycle "
            f"{cycle + 1}/{WALK_CYCLES} ---"
        )

        for frame_index, commands in enumerate(
            FORWARD_CYCLE,
            start=1,
        ):

            if not execute_group(
                v,
                commands,
                f"Frame {frame_index}",
            ):
                raise SystemExit

    gait_end_xyz = data.qpos[:3].copy()
    gait_end_time = data.time

    gait_delta = (
        gait_end_xyz
        - start_xyz
    )

    print("\n=== GAIT RESULT ===")

    print(
        f"Simulation duration: "
        f"{gait_end_time - start_time:.3f} s"
    )

    print(
        f"Start XYZ : {start_xyz}"
    )

    print(
        f"Gait end  : {gait_end_xyz}"
    )

    print(
        f"Delta XYZ : {gait_delta}"
    )

    print(
        f"Forward Δx: "
        f"{gait_delta[0]:.4f} m"
    )

    print(
        f"Lateral Δy: "
        f"{gait_delta[1]:.4f} m"
    )

    print("\n[4] Hold final pose")

    while v.is_running():
        simulate(v, 0.02)