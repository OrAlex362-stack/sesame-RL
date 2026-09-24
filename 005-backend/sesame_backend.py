from pathlib import Path
from collections import deque
import importlib.util
import math
import threading
import time

import mujoco
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import TD3


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.home() / "sesame-RL"

ENV_FILE = (
    ROOT
    / "004-command-conditioned-RL"
    / "01-command-env.py"
)

MODEL_FILE = (
    ROOT
    / "004-command-conditioned-RL"
    / "results"
    / "td3"
    / "best_model"
    / "best_model.zip"
)

TARGET_DT = 0.020
TARGET_HZ = 1.0 / TARGET_DT


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

spec = importlib.util.spec_from_file_location(
    "command_env",
    ENV_FILE,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SesameCommandEnv = module.SesameCommandEnv


env = SesameCommandEnv(
    render_mode=None,
    stop_probability=0.0,
)

model = TD3.load(
    MODEL_FILE,
    env=env,
    device="cpu",
)

obs, info = env.reset()


# ============================================================
# MUJOCO ROOT BODY
# ============================================================

mj_model = env.model
mj_data = env.data

free_joints = np.where(
    mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
)[0]

if len(free_joints) == 0:
    raise RuntimeError(
        "No MuJoCo free joint found for Sesame root body."
    )

root_joint_id = int(free_joints[0])
root_body_id = int(
    mj_model.jnt_bodyid[root_joint_id]
)

root_body_name = mujoco.mj_id2name(
    mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    root_body_id,
)

print(f"Root body: {root_body_name}")
print(f"Root body ID: {root_body_id}")


# ============================================================
# SHARED STATE
# ============================================================

lock = threading.Lock()

command = {
    "vx": 0.0,
    "yaw": 0.0,
}

reset_requested = False

telemetry = {
    "ready": False,
}

period_history = deque(maxlen=100)

total_steps = 0
episode = 0
deadline_misses = 0


# ============================================================
# HELPERS
# ============================================================

def quaternion_to_rpy(q):
    """
    MuJoCo quaternion format:
    [w, x, y, z]
    """

    w, x, y, z = q

    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = float(
        np.clip(sin_pitch, -1.0, 1.0)
    )

    pitch = math.asin(sin_pitch)

    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    return roll, pitch, yaw


def get_robot_pose():

    position = mj_data.xpos[root_body_id].copy()
    quaternion = mj_data.xquat[root_body_id].copy()

    roll, pitch, yaw = quaternion_to_rpy(
        quaternion
    )

    # MuJoCo spatial velocity:
    # [angular xyz, linear xyz]
    velocity = np.zeros(6)

    mujoco.mj_objectVelocity(
        mj_model,
        mj_data,
        mujoco.mjtObj.mjOBJ_BODY,
        root_body_id,
        velocity,
        0,  # world frame
    )

    angular = velocity[:3]
    linear = velocity[3:]

    return {
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },

        "orientation": {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),

            "quaternion": {
                "w": float(quaternion[0]),
                "x": float(quaternion[1]),
                "y": float(quaternion[2]),
                "z": float(quaternion[3]),
            },
        },

        "velocity": {
            "linear": {
                "x": float(linear[0]),
                "y": float(linear[1]),
                "z": float(linear[2]),
            },

            "angular": {
                "x": float(angular[0]),
                "y": float(angular[1]),
                "z": float(angular[2]),
            },
        },
    }


# ============================================================
# CONTROL LOOP
# ============================================================

def control_loop():

    global obs
    global reset_requested
    global telemetry
    global total_steps
    global episode
    global deadline_misses

    previous_start = None

    next_tick = time.perf_counter()

    while True:

        scheduled = next_tick

        now = time.perf_counter()

        if now < scheduled:
            time.sleep(scheduled - now)

        loop_start = time.perf_counter()


        # -----------------------------
        # Frequency measurement
        # -----------------------------

        if previous_start is not None:

            period = loop_start - previous_start
            period_history.append(period)

        previous_start = loop_start


        # -----------------------------
        # Copy command
        # -----------------------------

        with lock:

            vx = command["vx"]
            yaw = command["yaw"]

            do_reset = reset_requested

            if reset_requested:
                reset_requested = False


        # -----------------------------
        # Reset
        # -----------------------------

        if do_reset:

            obs, info = env.reset()

            episode += 1


        # -----------------------------
        # RL control
        # -----------------------------

        env.set_command(
            vx=vx,
            yaw=yaw,
        )

        compute_start = time.perf_counter()

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        compute_end = time.perf_counter()

        compute_ms = (
            compute_end - compute_start
        ) * 1000.0

        total_steps += 1


        # -----------------------------
        # Deadline
        # -----------------------------

        if compute_end > scheduled + TARGET_DT:
            deadline_misses += 1


        # -----------------------------
        # Actual Hz
        # -----------------------------

        if len(period_history) > 0:

            mean_period = sum(period_history) / len(
                period_history
            )

            actual_hz = 1.0 / mean_period

        else:
            actual_hz = 0.0


        # -----------------------------
        # Robot telemetry
        # -----------------------------

        robot = get_robot_pose()


        snapshot = {

            "sim": {
                "time": float(mj_data.time),
                "step": total_steps,
                "episode": episode,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },

            "command": {
                "vx": float(vx),
                "yaw": float(yaw),
            },

            "base": robot,

            "policy": {
                "name": "TD3",
                "action": [
                    float(x)
                    for x in np.asarray(action).flatten()
                ],
            },

            "controller": {
                "target_hz": TARGET_HZ,
                "actual_hz": float(actual_hz),
                "compute_ms": float(compute_ms),
                "deadline_misses": deadline_misses,
            },
        }


        with lock:
            telemetry = snapshot


        # -----------------------------
        # Episode handling
        # -----------------------------

        if terminated or truncated:

            obs, info = env.reset()
            episode += 1


        next_tick = scheduled + TARGET_DT


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="Sesame RL Backend",
)


class Command(BaseModel):
    vx: float = 0.0
    yaw: float = 0.0


@app.get("/")
def root():

    return {
        "name": "Sesame RL Backend",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/status")
def status():

    return {
        "backend": "running",
        "policy": "TD3",
        "model": MODEL_FILE.name,
        "root_body": root_body_name,
        "target_hz": TARGET_HZ,
    }


@app.get("/state")
def state():

    with lock:
        return dict(telemetry)


@app.post("/command")
def set_command(cmd: Command):

    with lock:

        command["vx"] = cmd.vx
        command["yaw"] = cmd.yaw

    return {
        "accepted": True,
        "vx": cmd.vx,
        "yaw": cmd.yaw,
    }


@app.post("/reset")
def reset():

    global reset_requested

    with lock:
        reset_requested = True

    return {
        "reset_requested": True,
    }

@app.get("/model/meshes")
def model_meshes():

    meshes = []

    for geom_id in range(mj_model.ngeom):

        # Only actual mesh geoms
        if (
            mj_model.geom_type[geom_id]
            != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue

        mesh_id = int(
            mj_model.geom_dataid[geom_id]
        )

        if mesh_id < 0:
            continue


        # ---------------------------------
        # Mesh vertices
        # ---------------------------------

        v_start = int(
            mj_model.mesh_vertadr[mesh_id]
        )

        v_count = int(
            mj_model.mesh_vertnum[mesh_id]
        )

        vertices = (
            mj_model.mesh_vert[
                v_start:v_start + v_count
            ]
            .copy()
        )


        # ---------------------------------
        # Mesh faces
        # ---------------------------------

        f_start = int(
            mj_model.mesh_faceadr[mesh_id]
        )

        f_count = int(
            mj_model.mesh_facenum[mesh_id]
        )

        faces = (
            mj_model.mesh_face[
                f_start:f_start + f_count
            ]
            .copy()
        )


        # ---------------------------------
        # Current geom world pose
        # ---------------------------------

        position = (
            mj_data.geom_xpos[geom_id]
            .copy()
        )

        rotation = (
            mj_data.geom_xmat[geom_id]
            .reshape(3, 3)
            .copy()
        )


        # ---------------------------------
        # Convert vertices directly to
        # MuJoCo world coordinates
        # ---------------------------------

        world_vertices = (
            vertices @ rotation.T
            + position
        )


        # ---------------------------------
        # Names
        # ---------------------------------

        geom_name = mujoco.mj_id2name(
            mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        body_id = int(
            mj_model.geom_bodyid[geom_id]
        )

        body_name = mujoco.mj_id2name(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )


        meshes.append({
            "geom_id": geom_id,
            "geom_name": geom_name,
            "body_id": body_id,
            "body_name": body_name,

            "vertices":
                world_vertices.tolist(),

            "faces":
                faces.tolist(),
        })


    return {
        "mesh_count": len(meshes),
        "meshes": meshes,
    }
# ============================================================
# START CONTROL THREAD
# ============================================================

threading.Thread(
    target=control_loop,
    daemon=True,
).start()