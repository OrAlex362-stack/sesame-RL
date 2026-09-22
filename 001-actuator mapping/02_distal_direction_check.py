import time
import numpy as np
import mujoco
from mujoco import viewer

from sesame_ml.model import load_model, make_data
from sesame_ml.constants import (
    JOINT_NAMES,
    STAND_ANGLES_RAD,
    JOINT_LIMITS_RAD,
)

model = load_model()
data = make_data(model, settle=True)

OFFSET_DEG = 8.0
offset = np.deg2rad(OFFSET_DEG)

# Only joints 4–7 need another test.
TEST_JOINTS = [4, 5, 6, 7]

# Move slightly away from mechanical limit first.
REFERENCE_DEG = {
    4: 15.0,    # R4: stand 0
    5: 165.0,   # R3: stand 180
    6: 15.0,    # L3: stand 0
    7: 165.0,   # L4: stand 180
}


def hold(v, targets, seconds):
    start = time.time()

    while v.is_running() and time.time() - start < seconds:

        targets = np.clip(
            targets,
            JOINT_LIMITS_RAD[:, 0],
            JOINT_LIMITS_RAD[:, 1],
        )

        data.ctrl[:] = targets
        mujoco.mj_step(model, data)
        v.sync()

        time.sleep(model.opt.timestep)


with viewer.launch_passive(model, data) as v:

    print("\n=== G1-C2 DISTAL JOINT DIRECTION CHECK ===")

    for idx in TEST_JOINTS:

        name = JOINT_NAMES[idx]
        ref = np.deg2rad(REFERENCE_DEG[idx])

        print("\n================================")
        print(f"INDEX {idx} / JOINT {name}")
        print(f"REFERENCE = {REFERENCE_DEG[idx]} deg")
        print("================================")

        # Reference pose
        targets = STAND_ANGLES_RAD.copy()
        targets[idx] = ref

        print("STATE: REFERENCE")
        hold(v, targets, 2.5)

        # Positive
        positive = targets.copy()
        positive[idx] = ref + offset

        print(f"STATE: +{OFFSET_DEG} deg")
        hold(v, positive, 2.5)

        # Back to reference
        print("STATE: REFERENCE")
        hold(v, targets, 1.5)

        # Negative
        negative = targets.copy()
        negative[idx] = ref - offset

        print(f"STATE: -{OFFSET_DEG} deg")
        hold(v, negative, 2.5)

        # Back to reference
        print("STATE: REFERENCE")
        hold(v, targets, 1.5)

        # Back to normal stand
        print("STATE: STAND")
        hold(v, STAND_ANGLES_RAD.copy(), 2.0)

    print("\n=== G1-C2 COMPLETE ===")

    while v.is_running():
        data.ctrl[:] = STAND_ANGLES_RAD
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(model.opt.timestep)