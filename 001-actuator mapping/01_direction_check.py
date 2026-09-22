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

OFFSET_DEG = 8.0
HOLD_SECONDS = 2.5

offset = np.deg2rad(OFFSET_DEG)

model = load_model()
data = make_data(model, settle=True)


def hold_pose(v, targets, duration):
    start = time.time()

    while v.is_running() and time.time() - start < duration:

        safe_targets = np.clip(
            targets,
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1],
        )

        data.ctrl[:] = safe_targets

        mujoco.mj_step(model, data)
        v.sync()

        time.sleep(model.opt.timestep)


with viewer.launch_passive(model, data) as v:

    print("\n=== G1-B/C DIRECTION CHECK ===")

    hold_pose(v, STAND_ANGLES_RAD.copy(), 2.0)

    for i, name in enumerate(JOINT_NAMES):

        if not v.is_running():
            break

        print("\n================================")
        print(f"INDEX {i} / JOINT {name}")
        print(f"STAND = {STAND_ANGLES_DEG[i]:.1f} deg")
        print("================================")

        # Stand
        print("STATE: STAND")
        hold_pose(
            v,
            STAND_ANGLES_RAD.copy(),
            HOLD_SECONDS,
        )

        # Positive offset
        positive = STAND_ANGLES_RAD.copy()
        positive[i] += offset

        print(f"STATE: +{OFFSET_DEG:.1f} deg")
        hold_pose(
            v,
            positive,
            HOLD_SECONDS,
        )

        # Stand
        print("STATE: STAND")
        hold_pose(
            v,
            STAND_ANGLES_RAD.copy(),
            1.5,
        )

        # Negative offset
        negative = STAND_ANGLES_RAD.copy()
        negative[i] -= offset

        print(f"STATE: -{OFFSET_DEG:.1f} deg")
        hold_pose(
            v,
            negative,
            HOLD_SECONDS,
        )

        # Stand again
        print("STATE: STAND")
        hold_pose(
            v,
            STAND_ANGLES_RAD.copy(),
            1.5,
        )

    print("\n=== DIRECTION CHECK COMPLETE ===")

    while v.is_running():

        data.ctrl[:] = STAND_ANGLES_RAD
        mujoco.mj_step(model, data)
        v.sync()

        time.sleep(model.opt.timestep)