import time

import mujoco
from mujoco import viewer

from sesame_ml.model import load_model, make_data
from sesame_ml.constants import STAND_ANGLES_RAD

# 1. Load Sesame MJCF + STL assets
model = load_model()

# 2. Create simulation data and put Sesame in stand pose
data = make_data(model, settle=True)

print("Sesame loaded successfully")
print(f"Actuators : {model.nu}")
print(f"Bodies    : {model.nbody}")
print(f"Geometries: {model.ngeom}")
print(f"Timestep  : {model.opt.timestep} s")

# 3. Open interactive MuJoCo viewer
with viewer.launch_passive(model, data) as v:

    while v.is_running():

        start = time.time()

        # Keep the 8 servos at the predefined standing pose
        data.ctrl[:] = STAND_ANGLES_RAD

        # Advance physics
        mujoco.mj_step(model, data)

        # Update viewer
        v.sync()

        # Approximately real-time simulation
        remaining = model.opt.timestep - (time.time() - start)

        if remaining > 0:
            time.sleep(remaining)