#!/usr/bin/env python3

"""
003-00 — Sesame RL Environment Baseline
=======================================

Purpose
-------
Convert the validated firmware/MuJoCo pipeline into a standard
Gymnasium reinforcement-learning environment.

This stage DOES NOT train PPO.

Validation targets
------------------
1. Gymnasium reset()/step()
2. Observation dimensions
3. Action dimensions
4. Action -> joint target scaling
5. Servo dynamics
6. Body-frame velocity correctness
7. Reward decomposition
8. Termination / truncation
9. Random-policy smoke test
10. Stable-Baselines3 check_env compatibility

Action
------
8D continuous action:

    action ∈ [-1, 1]^8

Converted into:

    q_target = q_stand + action * 0.580 rad


Observation
-----------
8   joint positions relative to stand
8   normalized joint velocities
3   body-frame linear velocity
3   body-frame angular velocity
3   projected gravity
8   previous action

Total = 33
"""

from pathlib import Path
import argparse
import csv

import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces

except ImportError:
    raise ImportError(
        "\nGymnasium is required.\n\n"
        "Install with:\n"
        "    pip install gymnasium\n"
    )


from sesame_ml.model import (
    load_model,
    make_data,
)

from sesame_ml.constants import (
    JOINT_INDEX,
    JOINT_NAMES,
    STAND_ANGLES_DEG,
    STAND_ANGLES_RAD,
    JOINT_LIMITS_RAD,
)


# ============================================================
# 1. SIMULATION CONFIG
# ============================================================

PHYSICS_DT = 0.002
CONTROL_DT = 0.020

SUBSTEPS = int(
    round(
        CONTROL_DT
        / PHYSICS_DT
    )
)

SERVO_TAU = 0.045

SERVO_MAX_SPEED_DEG = 600.0

SERVO_MAX_SPEED_RAD = np.deg2rad(
    SERVO_MAX_SPEED_DEG
)

ACTION_SCALE = 0.580


# ============================================================
# 2. EPISODE CONFIG
# ============================================================

EPISODE_DURATION = 20.0

MAX_EPISODE_STEPS = int(
    round(
        EPISODE_DURATION
        / CONTROL_DT
    )
)

RESET_SETTLE_DURATION = 1.0

RESET_SETTLE_STEPS = int(
    round(
        RESET_SETTLE_DURATION
        / PHYSICS_DT
    )
)


# ============================================================
# 3. TERMINATION
# ============================================================

UPRIGHT_THRESHOLD = 0.25

HEIGHT_FRACTION_THRESHOLD = 0.45


# ============================================================
# 4. INITIAL REWARD
#
# IMPORTANT:
# These are 003-00 validation weights.
# They are NOT final PPO reward weights.
# ============================================================

W_FORWARD = 2.0

W_LATERAL = 0.50

W_YAW = 0.05

W_UPRIGHT = 0.20

W_ACTION_RATE = 0.01

ALIVE_REWARD = 0.01

FALL_PENALTY = 10.0


# ============================================================
# 5. OUTPUT
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR.parent
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "003_rl_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STEP_CSV = (
    OUTPUT_DIR
    / "00_random_policy_steps.csv"
)

EPISODE_CSV = (
    OUTPUT_DIR
    / "00_random_policy_episodes.csv"
)


# ============================================================
# 6. BASIC HELPERS
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

        if name is None:
            return ""

        return name

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
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 7. JOINT LIMITS
# ============================================================

def clip_joint_limits(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    ).copy()

    limits = JOINT_LIMITS_RAD

    # --------------------------------------------------------
    # dictionary format
    # --------------------------------------------------------

    if isinstance(
        limits,
        dict,
    ):

        for name in JOINT_NAMES:

            if name not in limits:
                continue

            index = JOINT_INDEX[
                name
            ]

            low, high = (
                limits[name]
            )

            q[index] = np.clip(
                q[index],
                low,
                high,
            )

        return q

    # --------------------------------------------------------
    # array format
    # --------------------------------------------------------

    limits = np.asarray(
        limits
    )

    if (
        limits.ndim == 2
        and limits.shape[0] == len(q)
        and limits.shape[1] >= 2
    ):

        q = np.clip(
            q,
            limits[:, 0],
            limits[:, 1],
        )

    return q


# ============================================================
# 8. FLOATING BASE
# ============================================================

def find_free_joint(
    model,
):

    for joint_id in range(
        model.njnt
    ):

        if (
            model.jnt_type[
                joint_id
            ]
            == mujoco.mjtJoint.mjJNT_FREE
        ):

            body_id = int(
                model.jnt_bodyid[
                    joint_id
                ]
            )

            qpos_adr = int(
                model.jnt_qposadr[
                    joint_id
                ]
            )

            dof_adr = int(
                model.jnt_dofadr[
                    joint_id
                ]
            )

            return (
                joint_id,
                body_id,
                qpos_adr,
                dof_adr,
            )

    raise RuntimeError(
        "No floating-base free joint found."
    )


# ============================================================
# 9. JOINT MAP
# ============================================================

def build_joint_map(
    model,
):

    result = {}

    for name in JOINT_NAMES:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:

            raise RuntimeError(
                f"Joint not found: {name}"
            )

        result[name] = {

            "joint_id":
                joint_id,

            "qpos_adr":
                int(
                    model.jnt_qposadr[
                        joint_id
                    ]
                ),

            "dof_adr":
                int(
                    model.jnt_dofadr[
                        joint_id
                    ]
                ),
        }

    return result


# ============================================================
# 10. ACTUATOR MAP
# ============================================================

def build_actuator_map(
    model,
):

    result = {}

    for joint_name in JOINT_NAMES:

        # ----------------------------------------------------
        # Attempt 1:
        # actuator name == joint name
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Attempt 2:
        # inspect transmission joint
        # ----------------------------------------------------

        for actuator_id in range(
            model.nu
        ):

            joint_id = int(
                model.actuator_trnid[
                    actuator_id,
                    0,
                ]
            )

            if joint_id < 0:
                continue

            target_joint_name = safe_name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if (
                target_joint_name
                == joint_name
            ):

                result[
                    joint_name
                ] = actuator_id

                break

    missing = [
        name
        for name in JOINT_NAMES
        if name not in result
    ]

    if missing:

        raise RuntimeError(
            "Missing actuator mappings: "
            f"{missing}"
        )

    return result


# ============================================================
# 11. RL ENVIRONMENT
# ============================================================

class SesameRLEnv(
    gym.Env
):

    metadata = {
        "render_modes": [
            "human",
        ],

        "render_fps":
            int(
                round(
                    1.0
                    / CONTROL_DT
                )
            ),
    }

    def __init__(
        self,
        render_mode=None,
    ):

        super().__init__()

        if render_mode not in [
            None,
            "human",
        ]:

            raise ValueError(
                f"Unsupported render mode: "
                f"{render_mode}"
            )

        self.render_mode = (
            render_mode
        )

        # ====================================================
        # MODEL
        # ====================================================

        self.model = (
            load_model()
        )

        try:

            self.data = (
                make_data(
                    self.model
                )
            )

        except TypeError:

            self.data = (
                make_data()
            )

        self.model.opt.timestep = (
            PHYSICS_DT
        )

        # ====================================================
        # FLOATING BASE
        # ====================================================

        (
            self.free_joint_id,
            self.base_body_id,
            self.free_qpos_adr,
            self.free_dof_adr,
        ) = find_free_joint(
            self.model
        )

        # ====================================================
        # MAPS
        # ====================================================

        self.joint_map = (
            build_joint_map(
                self.model
            )
        )

        self.actuator_map = (
            build_actuator_map(
                self.model
            )
        )

        # ====================================================
        # STAND CONFIG
        # ====================================================

        self.stand_angles = (
            np.asarray(
                STAND_ANGLES_RAD,
                dtype=np.float64,
            ).copy()
        )

        # ====================================================
        # ACTION SPACE
        # ====================================================

        self.action_space = (
            spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    len(
                        JOINT_NAMES
                    ),
                ),
                dtype=np.float32,
            )
        )

        # ====================================================
        # OBSERVATION SPACE
        #
        # q relative          = 8
        # q velocity          = 8
        # body linear vel     = 3
        # body angular vel    = 3
        # gravity projection  = 3
        # previous action     = 8
        #
        # TOTAL               = 33
        # ====================================================

        self.obs_dim = (
            len(JOINT_NAMES)
            + len(JOINT_NAMES)
            + 3
            + 3
            + 3
            + len(JOINT_NAMES)
        )

        self.observation_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    self.obs_dim,
                ),
                dtype=np.float32,
            )
        )

        # ====================================================
        # INTERNAL STATE
        # ====================================================

        self.previous_action = (
            np.zeros(
                len(
                    JOINT_NAMES
                ),
                dtype=np.float64,
            )
        )

        self.servo_target = (
            self.stand_angles.copy()
        )

        self.episode_step = 0

        self.initial_height = None

        self.viewer = None


    # ========================================================
    # JOINT POSITION
    # ========================================================

    def _get_joint_positions(
        self,
    ):

        q = np.zeros(
            len(
                JOINT_NAMES
            ),
            dtype=np.float64,
        )

        for name in JOINT_NAMES:

            index = JOINT_INDEX[
                name
            ]

            qpos_adr = (
                self.joint_map[
                    name
                ][
                    "qpos_adr"
                ]
            )

            q[index] = (
                self.data.qpos[
                    qpos_adr
                ]
            )

        return q


    # ========================================================
    # JOINT VELOCITY
    # ========================================================

    def _get_joint_velocities(
        self,
    ):

        qd = np.zeros(
            len(
                JOINT_NAMES
            ),
            dtype=np.float64,
        )

        for name in JOINT_NAMES:

            index = JOINT_INDEX[
                name
            ]

            dof_adr = (
                self.joint_map[
                    name
                ][
                    "dof_adr"
                ]
            )

            qd[index] = (
                self.data.qvel[
                    dof_adr
                ]
            )

        return qd


    # ========================================================
    # BODY ROTATION MATRIX
    # ========================================================

    def _get_base_rotation(
        self,
    ):

        rotation = (
            self.data.xmat[
                self.base_body_id
            ].reshape(
                3,
                3,
            )
        )

        return rotation


    # ========================================================
    # UPRIGHT
    # ========================================================

    def _get_upright(
        self,
    ):

        rotation = (
            self._get_base_rotation()
        )

        # body Z dotted with world Z
        return float(
            rotation[2, 2]
        )


    # ========================================================
    # PROJECTED GRAVITY
    # ========================================================

    def _get_projected_gravity(
        self,
    ):

        rotation = (
            self._get_base_rotation()
        )

        gravity_world = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64,
        )

        # world -> body
        gravity_body = (
            rotation.T
            @ gravity_world
        )

        return gravity_body


    # ========================================================
    # BODY VELOCITY
    #
    # IMPORTANT FIX
    # ========================================================

    def _get_body_velocity(
        self,
    ):
        """
        Robust floating-base velocity extraction.

        Free-joint qvel layout:

            qvel[adr + 0 : adr + 3]
                translational velocity

            qvel[adr + 3 : adr + 6]
                rotational velocity

        Linear velocity is converted from world coordinates
        into the robot body frame.

        Angular velocity is taken directly from the free-joint
        rotational velocity.

        Returns
        -------
        linear_body : np.ndarray shape (3,)
            [forward, lateral, vertical]

        angular_body : np.ndarray shape (3,)
            [roll_rate, pitch_rate, yaw_rate]
        """

        adr = (
            self.free_dof_adr
        )

        # ----------------------------------------------------
        # floating-base linear velocity
        # ----------------------------------------------------

        linear_world = np.asarray(
            self.data.qvel[
                adr
                :
                adr + 3
            ],
            dtype=np.float64,
        ).copy()

        # ----------------------------------------------------
        # floating-base angular velocity
        # ----------------------------------------------------

        angular_body = np.asarray(
            self.data.qvel[
                adr + 3
                :
                adr + 6
            ],
            dtype=np.float64,
        ).copy()

        # ----------------------------------------------------
        # world -> body linear velocity
        # ----------------------------------------------------

        rotation = (
            self._get_base_rotation()
        )

        linear_body = (
            rotation.T
            @ linear_world
        )

        return (
            linear_body,
            angular_body,
        )


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_obs(
        self,
    ):

        q = (
            self._get_joint_positions()
        )

        qd = (
            self._get_joint_velocities()
        )

        (
            linear_velocity,
            angular_velocity,
        ) = self._get_body_velocity()

        gravity = (
            self._get_projected_gravity()
        )

        # ----------------------------------------------------
        # relative joint positions
        # ----------------------------------------------------

        q_relative = (
            q
            - self.stand_angles
        ) / ACTION_SCALE

        # ----------------------------------------------------
        # normalized joint velocities
        # ----------------------------------------------------

        qd_normalized = (
            qd
            / SERVO_MAX_SPEED_RAD
        )

        obs = np.concatenate(
            [
                q_relative,
                qd_normalized,
                linear_velocity,
                angular_velocity,
                gravity,
                self.previous_action,
            ]
        )

        return obs.astype(
            np.float32
        )


    # ========================================================
    # ACTION -> TARGET
    # ========================================================

    def _action_to_target(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        desired = (
            self.stand_angles
            + ACTION_SCALE
            * action
        )

        desired = (
            clip_joint_limits(
                desired
            )
        )

        return desired


    # ========================================================
    # SERVO
    # ========================================================

    def _servo_step(
        self,
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
            self.servo_target
            + alpha
            * (
                desired
                - self.servo_target
            )
        )

        max_delta = (
            SERVO_MAX_SPEED_RAD
            * CONTROL_DT
        )

        delta = np.clip(
            filtered
            - self.servo_target,
            -max_delta,
            max_delta,
        )

        self.servo_target = (
            self.servo_target
            + delta
        )


    # ========================================================
    # APPLY TARGET
    # ========================================================

    def _apply_servo_target(
        self,
    ):

        for name in JOINT_NAMES:

            index = JOINT_INDEX[
                name
            ]

            actuator_id = (
                self.actuator_map[
                    name
                ]
            )

            self.data.ctrl[
                actuator_id
            ] = (
                self.servo_target[
                    index
                ]
            )


    # ========================================================
    # REWARD
    # ========================================================

    def _compute_reward(
        self,
        action,
    ):

        (
            linear_velocity,
            angular_velocity,
        ) = self._get_body_velocity()

        # ----------------------------------------------------
        # BODY FRAME
        # ----------------------------------------------------

        forward_velocity = float(
            linear_velocity[0]
        )

        lateral_velocity = float(
            linear_velocity[1]
        )

        vertical_velocity = float(
            linear_velocity[2]
        )

        roll_rate = float(
            angular_velocity[0]
        )

        pitch_rate = float(
            angular_velocity[1]
        )

        yaw_rate = float(
            angular_velocity[2]
        )

        upright = (
            self._get_upright()
        )

        action_rate = (
            action
            - self.previous_action
        )

        # ----------------------------------------------------
        # COMPONENTS
        # ----------------------------------------------------

        reward_forward = (
            W_FORWARD
            * forward_velocity
        )

        penalty_lateral = (
            W_LATERAL
            * abs(
                lateral_velocity
            )
        )

        penalty_yaw = (
            W_YAW
            * abs(
                yaw_rate
            )
        )

        penalty_upright = (
            W_UPRIGHT
            * (
                1.0
                - upright
            ) ** 2
        )

        penalty_action_rate = (
            W_ACTION_RATE
            * float(
                np.mean(
                    action_rate ** 2
                )
            )
        )

        reward = (
            ALIVE_REWARD
            + reward_forward
            - penalty_lateral
            - penalty_yaw
            - penalty_upright
            - penalty_action_rate
        )

        components = {

            "reward_forward":
                float(
                    reward_forward
                ),

            "penalty_lateral":
                float(
                    penalty_lateral
                ),

            "penalty_yaw":
                float(
                    penalty_yaw
                ),

            "penalty_upright":
                float(
                    penalty_upright
                ),

            "penalty_action_rate":
                float(
                    penalty_action_rate
                ),

            "alive_reward":
                float(
                    ALIVE_REWARD
                ),

            "forward_velocity":
                forward_velocity,

            "lateral_velocity":
                lateral_velocity,

            "vertical_velocity":
                vertical_velocity,

            "roll_rate":
                roll_rate,

            "pitch_rate":
                pitch_rate,

            "yaw_rate":
                yaw_rate,

            "upright":
                float(
                    upright
                ),
        }

        return (
            float(
                reward
            ),
            components,
        )


    # ========================================================
    # TERMINATION
    # ========================================================

    def _check_termination(
        self,
    ):

        body_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        upright = (
            self._get_upright()
        )

        height_threshold = (
            self.initial_height
            * HEIGHT_FRACTION_THRESHOLD
        )

        fallen_height = bool(
            body_height
            < height_threshold
        )

        fallen_orientation = bool(
            upright
            < UPRIGHT_THRESHOLD
        )

        terminated = bool(
            fallen_height
            or fallen_orientation
        )

        return (
            terminated,
            body_height,
            upright,
            height_threshold,
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        # ----------------------------------------------------
        # reset MuJoCo
        # ----------------------------------------------------

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # ----------------------------------------------------
        # manually restore standing joint positions
        # ----------------------------------------------------

        for name in JOINT_NAMES:

            index = (
                JOINT_INDEX[
                    name
                ]
            )

            qpos_adr = (
                self.joint_map[
                    name
                ][
                    "qpos_adr"
                ]
            )

            self.data.qpos[
                qpos_adr
            ] = (
                self.stand_angles[
                    index
                ]
            )

        self.data.qvel[:] = 0.0

        self.previous_action = (
            np.zeros(
                len(
                    JOINT_NAMES
                ),
                dtype=np.float64,
            )
        )

        self.servo_target = (
            self.stand_angles.copy()
        )

        self.episode_step = 0

        # ----------------------------------------------------
        # apply stand control
        # ----------------------------------------------------

        self._apply_servo_target()

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # ----------------------------------------------------
        # settle
        # ----------------------------------------------------

        for _ in range(
            RESET_SETTLE_STEPS
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.initial_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        obs = (
            self._get_obs()
        )

        info = {

            "body_height":
                self.initial_height,

            "upright":
                self._get_upright(),

            "episode_step":
                0,
        }

        if (
            self.render_mode
            == "human"
        ):

            self.render()

        return (
            obs,
            info,
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        expected_shape = (
            len(
                JOINT_NAMES
            ),
        )

        if action.shape != expected_shape:

            raise ValueError(
                "Expected action shape "
                f"{expected_shape}, "
                f"received {action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # target
        # ----------------------------------------------------

        desired = (
            self._action_to_target(
                action
            )
        )

        # ----------------------------------------------------
        # servo
        # ----------------------------------------------------

        self._servo_step(
            desired
        )

        self._apply_servo_target()

        # ----------------------------------------------------
        # physics
        # ----------------------------------------------------

        for _ in range(
            SUBSTEPS
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.episode_step += 1

        # ----------------------------------------------------
        # reward
        # ----------------------------------------------------

        (
            reward,
            reward_info,
        ) = self._compute_reward(
            action
        )

        # ----------------------------------------------------
        # termination
        # ----------------------------------------------------

        (
            terminated,
            body_height,
            upright,
            height_threshold,
        ) = self._check_termination()

        truncated = bool(
            self.episode_step
            >= MAX_EPISODE_STEPS
        )

        if terminated:

            reward -= (
                FALL_PENALTY
            )

        # ----------------------------------------------------
        # previous action
        # ----------------------------------------------------

        self.previous_action = (
            action.copy()
        )

        # ----------------------------------------------------
        # observation
        # ----------------------------------------------------

        obs = (
            self._get_obs()
        )

        info = {

            **reward_info,

            "body_height":
                float(
                    body_height
                ),

            "height_threshold":
                float(
                    height_threshold
                ),

            "episode_step":
                int(
                    self.episode_step
                ),

            "terminated":
                bool(
                    terminated
                ),

            "truncated":
                bool(
                    truncated
                ),

            "reward_total":
                float(
                    reward
                ),
        }

        if (
            self.render_mode
            == "human"
        ):

            self.render()

        return (
            obs,
            float(
                reward
            ),
            terminated,
            truncated,
            info,
        )


    # ========================================================
    # RENDER
    # ========================================================

    def render(
        self,
    ):

        if (
            self.render_mode
            != "human"
        ):
            return

        if self.viewer is None:

            import mujoco.viewer

            self.viewer = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
            )

        if self.viewer.is_running():

            self.viewer.sync()


    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self,
    ):

        if self.viewer is None:
            return

        try:

            self.viewer.close()

        except Exception:

            pass

        self.viewer = None


# ============================================================
# 12. ENVIRONMENT CHECK
# ============================================================

def check_environment():

    print()
    print(
        "=== 003-00 ENVIRONMENT CHECK ==="
    )

    env = SesameRLEnv(
        render_mode=None
    )

    obs, info = env.reset(
        seed=1000
    )

    print(
        f"Observation shape : "
        f"{obs.shape}"
    )

    print(
        f"Expected shape    : "
        f"{env.observation_space.shape}"
    )

    print(
        f"Action shape      : "
        f"{env.action_space.shape}"
    )

    print(
        f"Initial height    : "
        f"{info['body_height']:.5f} m"
    )

    print(
        f"Initial upright   : "
        f"{info['upright']:.5f}"
    )

    observation_valid = (
        env.observation_space.contains(
            obs
        )
    )

    print(
        f"Observation valid : "
        f"{'PASS' if observation_valid else 'FAIL'}"
    )

    # ========================================================
    # ZERO ACTION
    # ========================================================

    zero_action = np.zeros(
        len(
            JOINT_NAMES
        ),
        dtype=np.float32,
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        step_info,
    ) = env.step(
        zero_action
    )

    zero_finite = bool(
        np.all(
            np.isfinite(
                obs
            )
        )
    )

    print(
        f"Zero-action step  : "
        f"{'PASS' if zero_finite else 'FAIL'}"
    )

    print(
        f"Zero reward       : "
        f"{reward:+.6f}"
    )

    print(
        f"Zero forward vel  : "
        f"{step_info['forward_velocity']:+.6f} m/s"
    )

    print(
        f"Zero lateral vel  : "
        f"{step_info['lateral_velocity']:+.6f} m/s"
    )

    # ========================================================
    # VELOCITY SANITY CHECK
    # ========================================================

    print()
    print(
        "--- VELOCITY SANITY CHECK ---"
    )

    (
        linear_velocity,
        angular_velocity,
    ) = env._get_body_velocity()

    print(
        "Body linear velocity : "
        f"["
        f"{linear_velocity[0]:+.6f}, "
        f"{linear_velocity[1]:+.6f}, "
        f"{linear_velocity[2]:+.6f}"
        f"] m/s"
    )

    print(
        "Body angular velocity: "
        f"["
        f"{np.rad2deg(angular_velocity[0]):+.3f}, "
        f"{np.rad2deg(angular_velocity[1]):+.3f}, "
        f"{np.rad2deg(angular_velocity[2]):+.3f}"
        f"] deg/s"
    )

    # ========================================================
    # SMALL CONTROLLED MOTION SANITY TEST
    # ========================================================

    print()
    print(
        "--- SMALL ACTION SANITY CHECK ---"
    )

    small_action = np.zeros(
        len(
            JOINT_NAMES
        ),
        dtype=np.float32,
    )

    # small symmetric perturbation
    small_action[:] = 0.05

    (
        obs,
        reward,
        terminated,
        truncated,
        small_info,
    ) = env.step(
        small_action
    )

    print(
        f"Forward velocity : "
        f"{small_info['forward_velocity']:+.6f} m/s"
    )

    print(
        f"Lateral velocity : "
        f"{small_info['lateral_velocity']:+.6f} m/s"
    )

    print(
        f"Vertical velocity: "
        f"{small_info['vertical_velocity']:+.6f} m/s"
    )

    print(
        f"Roll rate        : "
        f"{np.rad2deg(small_info['roll_rate']):+.3f} deg/s"
    )

    print(
        f"Pitch rate       : "
        f"{np.rad2deg(small_info['pitch_rate']):+.3f} deg/s"
    )

    print(
        f"Yaw rate         : "
        f"{np.rad2deg(small_info['yaw_rate']):+.3f} deg/s"
    )

    env.close()

    # ========================================================
    # SB3 CHECK
    # ========================================================

    try:

        from stable_baselines3.common.env_checker import (
            check_env,
        )

        env = SesameRLEnv(
            render_mode=None
        )

        check_env(
            env,
            warn=True,
        )

        env.close()

        print(
            "SB3 check_env     : PASS"
        )

    except ImportError:

        print(
            "SB3 check_env     : SKIPPED "
            "(stable-baselines3 not installed)"
        )

    except Exception as exc:

        print(
            "SB3 check_env     : FAIL"
        )

        print(
            f"Reason            : {exc}"
        )


# ============================================================
# 13. RANDOM POLICY VALIDATION
# ============================================================

def run_random_policy(
    episodes,
    render=False,
    seed=1000,
):

    print()
    print(
        "=== 003-00 RANDOM POLICY VALIDATION ==="
    )

    print(
        f"Episodes          : "
        f"{episodes}"
    )

    print(
        f"Episode steps     : "
        f"{MAX_EPISODE_STEPS}"
    )

    print(
        f"Episode duration  : "
        f"{EPISODE_DURATION:.1f} s"
    )

    env = SesameRLEnv(
        render_mode=(
            "human"
            if render
            else None
        )
    )

    rng = np.random.default_rng(
        seed
    )

    step_rows = []

    episode_rows = []

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(
        episodes
    ):

        obs, reset_info = env.reset(
            seed=(
                seed
                + episode
            )
        )

        total_reward = 0.0

        forward_values = []

        lateral_values = []

        vertical_values = []

        roll_values = []

        pitch_values = []

        yaw_values = []

        upright_values = []

        height_values = []

        terminated = False

        truncated = False

        final_step = 0

        # ====================================================
        # ROLLOUT
        # ====================================================

        for step in range(
            MAX_EPISODE_STEPS
        ):

            # ------------------------------------------------
            # Conservative random smoke-test action.
            #
            # PPO later gets full [-1, 1].
            # This stage only checks pipeline semantics.
            # ------------------------------------------------

            action = rng.uniform(
                low=-0.35,
                high=0.35,
                size=(
                    len(
                        JOINT_NAMES
                    ),
                ),
            ).astype(
                np.float32
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += (
                reward
            )

            forward_values.append(
                info[
                    "forward_velocity"
                ]
            )

            lateral_values.append(
                info[
                    "lateral_velocity"
                ]
            )

            vertical_values.append(
                info[
                    "vertical_velocity"
                ]
            )

            roll_values.append(
                info[
                    "roll_rate"
                ]
            )

            pitch_values.append(
                info[
                    "pitch_rate"
                ]
            )

            yaw_values.append(
                info[
                    "yaw_rate"
                ]
            )

            upright_values.append(
                info[
                    "upright"
                ]
            )

            height_values.append(
                info[
                    "body_height"
                ]
            )

            # ------------------------------------------------
            # STEP LOG
            # ------------------------------------------------

            row = {

                "episode":
                    episode,

                "step":
                    step,

                "time":
                    step
                    * CONTROL_DT,

                "reward":
                    float(
                        reward
                    ),

                "reward_forward":
                    info[
                        "reward_forward"
                    ],

                "penalty_lateral":
                    info[
                        "penalty_lateral"
                    ],

                "penalty_yaw":
                    info[
                        "penalty_yaw"
                    ],

                "penalty_upright":
                    info[
                        "penalty_upright"
                    ],

                "penalty_action_rate":
                    info[
                        "penalty_action_rate"
                    ],

                "forward_velocity":
                    info[
                        "forward_velocity"
                    ],

                "lateral_velocity":
                    info[
                        "lateral_velocity"
                    ],

                "vertical_velocity":
                    info[
                        "vertical_velocity"
                    ],

                "roll_rate":
                    info[
                        "roll_rate"
                    ],

                "pitch_rate":
                    info[
                        "pitch_rate"
                    ],

                "yaw_rate":
                    info[
                        "yaw_rate"
                    ],

                "upright":
                    info[
                        "upright"
                    ],

                "body_height":
                    info[
                        "body_height"
                    ],

                "terminated":
                    int(
                        terminated
                    ),

                "truncated":
                    int(
                        truncated
                    ),
            }

            step_rows.append(
                row
            )

            final_step = step

            if (
                terminated
                or truncated
            ):

                break

        # ====================================================
        # EPISODE SUMMARY
        # ====================================================

        episode_duration = (
            (
                final_step
                + 1
            )
            * CONTROL_DT
        )

        mean_forward = float(
            np.mean(
                forward_values
            )
        )

        mean_abs_forward = float(
            np.mean(
                np.abs(
                    forward_values
                )
            )
        )

        mean_lateral = float(
            np.mean(
                lateral_values
            )
        )

        mean_abs_lateral = float(
            np.mean(
                np.abs(
                    lateral_values
                )
            )
        )

        mean_abs_vertical = float(
            np.mean(
                np.abs(
                    vertical_values
                )
            )
        )

        mean_abs_roll = float(
            np.mean(
                np.abs(
                    roll_values
                )
            )
        )

        mean_abs_pitch = float(
            np.mean(
                np.abs(
                    pitch_values
                )
            )
        )

        mean_abs_yaw = float(
            np.mean(
                np.abs(
                    yaw_values
                )
            )
        )

        minimum_upright = float(
            np.min(
                upright_values
            )
        )

        minimum_height = float(
            np.min(
                height_values
            )
        )

        episode_rows.append({

            "episode":
                episode,

            "steps":
                final_step + 1,

            "duration":
                episode_duration,

            "total_reward":
                total_reward,

            "mean_forward_velocity":
                mean_forward,

            "mean_abs_forward_velocity":
                mean_abs_forward,

            "mean_lateral_velocity":
                mean_lateral,

            "mean_abs_lateral_velocity":
                mean_abs_lateral,

            "mean_abs_vertical_velocity":
                mean_abs_vertical,

            "mean_abs_roll_rate":
                mean_abs_roll,

            "mean_abs_pitch_rate":
                mean_abs_pitch,

            "mean_abs_yaw_rate":
                mean_abs_yaw,

            "minimum_upright":
                minimum_upright,

            "minimum_height":
                minimum_height,

            "terminated":
                int(
                    terminated
                ),

            "truncated":
                int(
                    truncated
                ),
        })

        # ====================================================
        # TERMINAL
        # ====================================================

        print()
        print(
            f"Episode {episode:02d}"
        )

        print(
            f"  steps         : "
            f"{final_step + 1}"
        )

        print(
            f"  reward        : "
            f"{total_reward:+.3f}"
        )

        print(
            f"  mean forward  : "
            f"{mean_forward:+.4f} m/s"
        )

        print(
            f"  mean |forward|: "
            f"{mean_abs_forward:.4f} m/s"
        )

        print(
            f"  mean lateral  : "
            f"{mean_lateral:+.4f} m/s"
        )

        print(
            f"  mean |lat|    : "
            f"{mean_abs_lateral:.4f} m/s"
        )

        print(
            f"  mean |vertical|: "
            f"{mean_abs_vertical:.4f} m/s"
        )

        print(
            f"  mean |roll|   : "
            f"{np.rad2deg(mean_abs_roll):.2f} deg/s"
        )

        print(
            f"  mean |pitch|  : "
            f"{np.rad2deg(mean_abs_pitch):.2f} deg/s"
        )

        print(
            f"  mean |yaw|    : "
            f"{np.rad2deg(mean_abs_yaw):.2f} deg/s"
        )

        print(
            f"  min upright   : "
            f"{minimum_upright:.4f}"
        )

        print(
            f"  min height    : "
            f"{minimum_height:.5f} m"
        )

        print(
            f"  terminated    : "
            f"{terminated}"
        )

        print(
            f"  truncated     : "
            f"{truncated}"
        )

    # ========================================================
    # CLOSE
    # ========================================================

    env.close()

    # ========================================================
    # EXPORT
    # ========================================================

    save_csv(
        STEP_CSV,
        step_rows,
    )

    save_csv(
        EPISODE_CSV,
        episode_rows,
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    rewards = np.asarray([
        r[
            "total_reward"
        ]
        for r in episode_rows
    ])

    forward = np.asarray([
        r[
            "mean_forward_velocity"
        ]
        for r in episode_rows
    ])

    forward_abs = np.asarray([
        r[
            "mean_abs_forward_velocity"
        ]
        for r in episode_rows
    ])

    lateral = np.asarray([
        r[
            "mean_abs_lateral_velocity"
        ]
        for r in episode_rows
    ])

    vertical = np.asarray([
        r[
            "mean_abs_vertical_velocity"
        ]
        for r in episode_rows
    ])

    roll = np.asarray([
        r[
            "mean_abs_roll_rate"
        ]
        for r in episode_rows
    ])

    pitch = np.asarray([
        r[
            "mean_abs_pitch_rate"
        ]
        for r in episode_rows
    ])

    yaw = np.asarray([
        r[
            "mean_abs_yaw_rate"
        ]
        for r in episode_rows
    ])

    terminated_count = sum(
        r[
            "terminated"
        ]
        for r in episode_rows
    )

    print()
    print("=" * 70)

    print(
        "003-00 RANDOM POLICY SUMMARY"
    )

    print("=" * 70)

    print(
        f"Episodes                    : "
        f"{len(episode_rows)}"
    )

    print(
        f"Mean total reward           : "
        f"{np.mean(rewards):+.3f}"
    )

    print(
        f"Mean forward velocity       : "
        f"{np.mean(forward):+.5f} m/s"
    )

    print(
        f"Mean |forward velocity|     : "
        f"{np.mean(forward_abs):.5f} m/s"
    )

    print(
        f"Mean |lateral velocity|     : "
        f"{np.mean(lateral):.5f} m/s"
    )

    print(
        f"Mean |vertical velocity|    : "
        f"{np.mean(vertical):.5f} m/s"
    )

    print(
        f"Mean |roll rate|            : "
        f"{np.rad2deg(np.mean(roll)):.3f} deg/s"
    )

    print(
        f"Mean |pitch rate|           : "
        f"{np.rad2deg(np.mean(pitch)):.3f} deg/s"
    )

    print(
        f"Mean |yaw rate|             : "
        f"{np.rad2deg(np.mean(yaw)):.3f} deg/s"
    )

    print(
        f"Falls / terminations        : "
        f"{terminated_count}"
        f"/{len(episode_rows)}"
    )

    print(
        f"Step CSV                    : "
        f"{STEP_CSV}"
    )

    print(
        f"Episode CSV                 : "
        f"{EPISODE_CSV}"
    )

    print("=" * 70)


# ============================================================
# 14. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
    )

    args = parser.parse_args()

    print()
    print("=" * 66)

    print(
        "00 / 003-00 SESAME RL ENVIRONMENT BASELINE"
    )

    print("=" * 66)

    print(
        f"Physics dt        : "
        f"{PHYSICS_DT:.4f} s"
    )

    print(
        f"Control dt        : "
        f"{CONTROL_DT:.4f} s"
    )

    print(
        f"Substeps          : "
        f"{SUBSTEPS}"
    )

    print(
        f"Servo tau         : "
        f"{SERVO_TAU:.3f} s"
    )

    print(
        f"Servo max speed   : "
        f"{SERVO_MAX_SPEED_DEG:.1f} deg/s"
    )

    print(
        f"Action scale      : "
        f"±{ACTION_SCALE:.3f} rad"
    )

    print(
        f"Observation dim   : "
        f"33"
    )

    print(
        f"Action dim        : "
        f"{len(JOINT_NAMES)}"
    )

    print(
        f"Episode duration  : "
        f"{EPISODE_DURATION:.1f} s"
    )

    print(
        f"Episode steps     : "
        f"{MAX_EPISODE_STEPS}"
    )

    # ========================================================
    # CHECK
    # ========================================================

    if not args.skip_check:

        check_environment()

    # ========================================================
    # RANDOM POLICY
    # ========================================================

    run_random_policy(
        episodes=args.episodes,
        render=args.render,
        seed=args.seed,
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()