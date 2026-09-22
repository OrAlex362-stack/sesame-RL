import time
import numpy as np
import mujoco
from mujoco import viewer

from sesame_ml.model import load_model, make_data
from sesame_ml.constants import (
    JOINT_NAMES,
    STAND_ANGLES_RAD,
    STAND_ANGLES_DEG,
    JOINT_LIMITS_RAD,
)


# -------------------------------------------------
# G1 settings
# -------------------------------------------------

AMPLITUDE_DEG = 8.0
MOVE_DURATION = 3.0
PAUSE_DURATION = 1.0

amplitude_rad = np.deg2rad(AMPLITUDE_DEG)


# -------------------------------------------------
# Load model
# -------------------------------------------------

model = load_model()
data = make_data(model, settle=True)

print("\n=== G1 ACTUATOR MAPPING CHECK ===\n")

print("Canonical joint order:")

for i, name in enumerate(JOINT_NAMES):
    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(
        f"{i}: joint={name:>2} "
        f" actuator={actuator_name:>10} "
        f" stand={STAND_ANGLES_DEG[i]:6.1f} deg"
    )

print("\nOnly ONE joint will move at a time.")
print(f"Motion amplitude: +/- {AMPLITUDE_DEG} deg\n")


# -------------------------------------------------
# Helper
# -------------------------------------------------

def run_for_seconds(v, duration, target_function):
    start = time.time()

    while v.is_running():

        elapsed = time.time() - start

        if elapsed >= duration:
            break

        targets = target_function(elapsed)

        # Respect physical limits
        targets = np.clip(
            targets,
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1]
        )

        data.ctrl[:] = targets

        mujoco.mj_step(model, data)

        v.sync()

        time.sleep(model.opt.timestep)


def stand(v, duration):
    def target(_):
        return STAND_ANGLES_RAD.copy()

    run_for_seconds(v, duration, target)


# -------------------------------------------------
# Viewer
# -------------------------------------------------

with viewer.launch_passive(model, data) as v:

    # initial stand
    print("Initial stand...")
    stand(v, 2.0)

    for joint_index, joint_name in enumerate(JOINT_NAMES):

        if not v.is_running():
            break

        print("\n--------------------------------")
        print(f"Testing INDEX {joint_index}")
        print(f"Joint: {joint_name}")
        print("--------------------------------")

        # sinusoidal motion around stand pose
        def move_joint(t, idx=joint_index):

            targets = STAND_ANGLES_RAD.copy()

            phase = (
                2.0
                * np.pi
                * t
                / MOVE_DURATION
            )

            targets[idx] += (
                amplitude_rad
                * np.sin(phase)
            )

            return targets

        run_for_seconds(
            v,
            MOVE_DURATION,
            move_joint
        )

        # return to stand before next joint
        stand(v, PAUSE_DURATION)

    print("\n=== G1 COMPLETE ===")

    # Stay standing until viewer closes
    while v.is_running():

        data.ctrl[:] = STAND_ANGLES_RAD

        mujoco.mj_step(model, data)

        v.sync()

        time.sleep(model.opt.timestep)