#!/usr/bin/env python3

"""
Sesame RL Environment
=====================

Shared Gymnasium environment for Sesame-RL.

Validated baseline:
- Physics dt       : 0.002 s
- Control dt       : 0.020 s
- Substeps         : 10
- Servo tau        : 0.045 s
- Servo max speed  : 600 deg/s
- Action scale     : ±0.580 rad
- Observation dim  : 33
- Action dim       : 8
- Episode length   : 1000 control steps

Observation
-----------
8   joint position relative to stand
8   normalized joint velocity
3   body-frame linear velocity
3   body-frame angular velocity
3   projected gravity
8   previous action

Total = 33

Action
------
8D continuous action:

    action ∈ [-1, 1]^8

Converted to joint target:

    q_target = q_stand + action * ACTION_SCALE
"""

import numpy as np
import mujoco
import gymnasium as gym

from gymnasium import spaces

from sesame_ml.model import (
    load_model,
    make_data,
)

from sesame_ml.constants import (
    JOINT_INDEX,
    JOINT_NAMES,
    STAND_ANGLES_RAD,
    JOINT_LIMITS_RAD,
)


# ============================================================
# SIMULATION
# ============================================================

PHYSICS_DT = 0.002
CONTROL_DT = 0.020

SUBSTEPS = int(
    round(
        CONTROL_DT
        / PHYSICS_DT
    )
)


# ============================================================
# SERVO
# ============================================================

SERVO_TAU = 0.045

SERVO_MAX_SPEED_DEG = 600.0

SERVO_MAX_SPEED_RAD = np.deg2rad(
    SERVO_MAX_SPEED_DEG
)

ACTION_SCALE = 0.580


# ============================================================
# EPISODE
# ============================================================

EPISODE_DURATION = 20.0

EPISODE_STEPS = int(
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
# TERMINATION
# ============================================================

UPRIGHT_THRESHOLD = 0.25

HEIGHT_THRESHOLD_RATIO = 0.45


# ============================================================
# REWARD V1 DEFAULT
# ============================================================

W_FORWARD = 2.0

W_LATERAL = 0.50

W_YAW = 0.05

W_UPRIGHT = 0.20

W_ACTION_RATE = 0.01

ALIVE_REWARD = 0.01

FALL_PENALTY = 10.0


# ============================================================
# HELPERS
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

        return (
            name
            if name is not None
            else ""
        )

    except Exception:

        return ""


# ============================================================
# JOINT LIMITS
# ============================================================

def clip_joint_limits(q):
    """
    Support either dictionary-style or array-style
    JOINT_LIMITS_RAD.
    """

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

            i = JOINT_INDEX[
                name
            ]

            low, high = (
                limits[name]
            )

            q[i] = np.clip(
                q[i],
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
# FLOATING BASE
# ============================================================

def find_free_joint(model):
    """
    Returns:
        joint_id
        base_body_id
        free_qpos_address
        free_dof_address
    """

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
# JOINT MAP
# ============================================================

def build_joint_map(model):

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
# ACTUATOR MAP
# ============================================================

def build_actuator_map(model):

    result = {}

    for joint_name in JOINT_NAMES:

        # ----------------------------------------------------
        # Strategy 1:
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
        # Strategy 2:
        # search transmission target joint
        # ----------------------------------------------------

        for aid in range(
            model.nu
        ):

            joint_id = int(
                model.actuator_trnid[
                    aid,
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
                ] = aid

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
# ENVIRONMENT
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


    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        render_mode=None,
        w_yaw=W_YAW,
    ):

        super().__init__()

        if render_mode not in [
            None,
            "human",
        ]:

            raise ValueError(
                f"Unsupported render_mode: "
                f"{render_mode}"
            )

        self.render_mode = (
            render_mode
        )

        # Reward parameter used for yaw ablation
        self.w_yaw = float(
            w_yaw
        )

        # ====================================================
        # LOAD MODEL
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
        # STAND CONFIGURATION
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
        # 8 q relative
        # 8 qdot normalized
        # 3 linear velocity
        # 3 angular velocity
        # 3 projected gravity
        # 8 previous action
        #
        # total = 33
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

        self.step_count = 0

        self.initial_height = 0.0

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

            i = JOINT_INDEX[
                name
            ]

            qpos_adr = (
                self.joint_map[
                    name
                ][
                    "qpos_adr"
                ]
            )

            q[i] = (
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

            i = JOINT_INDEX[
                name
            ]

            dof_adr = (
                self.joint_map[
                    name
                ][
                    "dof_adr"
                ]
            )

            qd[i] = (
                self.data.qvel[
                    dof_adr
                ]
            )

        return qd


    # ========================================================
    # BASE ROTATION
    # ========================================================

    def _get_base_rotation(
        self,
    ):

        return (
            self.data.xmat[
                self.base_body_id
            ].reshape(
                3,
                3,
            )
        )


    # ========================================================
    # UPRIGHT
    # ========================================================

    def _get_upright(
        self,
    ):

        rotation = (
            self._get_base_rotation()
        )

        return float(
            rotation[
                2,
                2,
            ]
        )


    # ========================================================
    # PROJECTED GRAVITY
    # ========================================================

    def _get_projected_gravity(
        self,
    ):

        gravity_world = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64,
        )

        rotation = (
            self._get_base_rotation()
        )

        gravity_body = (
            rotation.T
            @ gravity_world
        )

        return gravity_body


    # ========================================================
    # BODY VELOCITY
    # ========================================================

    def _get_body_velocity(
        self,
    ):
        """
        Free-joint qvel:

        first 3:
            global-frame linear velocity

        next 3:
            body-frame angular velocity

        Linear velocity is rotated into body coordinates.

        Returns:
            linear_body
                [forward, lateral, vertical]

            angular_body
                [roll_rate, pitch_rate, yaw_rate]
        """

        adr = (
            self.free_dof_adr
        )

        # ----------------------------------------------------
        # global linear velocity
        # ----------------------------------------------------

        linear_world = np.asarray(
            self.data.qvel[
                adr:
                adr + 3
            ],
            dtype=np.float64,
        ).copy()

        # ----------------------------------------------------
        # local angular velocity
        # ----------------------------------------------------

        angular_body = np.asarray(
            self.data.qvel[
                adr + 3:
                adr + 6
            ],
            dtype=np.float64,
        ).copy()

        rotation = (
            self._get_base_rotation()
        )

        # world -> body
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
        ) = (
            self._get_body_velocity()
        )

        gravity = (
            self._get_projected_gravity()
        )

        # ----------------------------------------------------
        # normalize joint positions relative to stand
        # ----------------------------------------------------

        q_relative = (
            q
            - self.stand_angles
        ) / ACTION_SCALE

        # ----------------------------------------------------
        # normalize joint velocity
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

        target = (
            self.stand_angles
            + ACTION_SCALE
            * action
        )

        target = (
            clip_joint_limits(
                target
            )
        )

        return target


    # ========================================================
    # SERVO MODEL
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
    # APPLY CONTROL
    # ========================================================

    def _apply_servo_target(
        self,
    ):

        for name in JOINT_NAMES:

            i = JOINT_INDEX[
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
                    i
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
        ) = (
            self._get_body_velocity()
        )

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

        action_delta = (
            action
            - self.previous_action
        )

        # ----------------------------------------------------
        # reward terms
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
            self.w_yaw
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
                    action_delta ** 2
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

        info = {

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

            "w_yaw":
                self.w_yaw,
        }

        return (
            float(
                reward
            ),
            info,
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
            * HEIGHT_THRESHOLD_RATIO
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

        if seed is not None:

            self.action_space.seed(
                seed
            )

        # ----------------------------------------------------
        # reset MuJoCo
        # ----------------------------------------------------

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # ----------------------------------------------------
        # set joint stand configuration
        # ----------------------------------------------------

        for name in JOINT_NAMES:

            i = JOINT_INDEX[
                name
            ]

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
                    i
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

        self.step_count = 0

        # ----------------------------------------------------
        # apply initial stand
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

            "w_yaw":
                self.w_yaw,
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

        if (
            action.shape
            != expected_shape
        ):

            raise ValueError(
                f"Expected action shape "
                f"{expected_shape}, "
                f"got {action.shape}"
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # action -> desired joint target
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
        # MuJoCo physics
        # ----------------------------------------------------

        for _ in range(
            SUBSTEPS
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.step_count += 1

        # ----------------------------------------------------
        # reward
        # ----------------------------------------------------

        (
            reward,
            reward_info,
        ) = (
            self._compute_reward(
                action
            )
        )

        # ----------------------------------------------------
        # termination
        # ----------------------------------------------------

        (
            terminated,
            body_height,
            upright,
            height_threshold,
        ) = (
            self._check_termination()
        )

        truncated = bool(
            self.step_count
            >= EPISODE_STEPS
        )

        if terminated:

            reward -= (
                FALL_PENALTY
            )

        # ----------------------------------------------------
        # save previous action
        # ----------------------------------------------------

        self.previous_action = (
            action.copy()
        )

        # ----------------------------------------------------
        # observation after action
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
                    self.step_count
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