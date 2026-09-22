import time
import numpy as np
import mujoco
from mujoco import viewer

from sesame_ml.model import load_model, make_data
from sesame_ml.constants import STAND_ANGLES_RAD


model = load_model()
data = make_data(model, settle=True)

print("Number of actuators:", model.nu)

for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )
    print(i, name)


with viewer.launch_passive(model, data) as v:

    t = 0.0

    while v.is_running():

        # Default = stand pose
        targets = STAND_ANGLES_RAD.copy()

        # Only move actuator 0 for now
        targets[0] += 0.25 * np.sin(2.0 * np.pi * 0.5 * t)
        targets[1] += 0.25 * np.sin(2.0 * np.pi * 0.5 * t)
        targets[2] += 0.25 * np.sin(2.0 * np.pi * 0.5 * t)
        targets[3] += 0.25 * np.sin(6.0 * np.pi * 0.5 * t)
        targets[4] += 0.25 * np.sin(2.0 * np.pi * 0.5 * t)
        targets[5] += 0.25 * np.sin(2.0 * np.pi * 0.5 * t)
        targets[6] += 0.25 * np.sin(5.0 * np.pi * 0.5 * t)
        targets[7] += 0.85 * np.sin(2.0 * np.pi * 0.5 * t)

        data.ctrl[:] = targets

        mujoco.mj_step(model, data)

        v.sync()

        t += model.opt.timestep

        time.sleep(model.opt.timestep)