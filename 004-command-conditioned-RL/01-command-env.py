#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
from gymnasium import spaces

# 003-RL is intentionally reused as the single source of robot dynamics/safety.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "003-RL"))

from sesame_rl_env import (  # noqa: E402
    ALIVE_REWARD,
    FALL_PENALTY,
    W_ACTION_RATE,
    W_FORWARD,
    W_LATERAL,
    W_UPRIGHT,
    SesameRLEnv,
)

MAX_VX = 0.20
MAX_YAW = 1.0
STOP_PROBABILITY = 0.15

# Errors equal to one command range retain exp(-1) of the tracking reward.
K_V = 1.0 / MAX_VX**2
K_YAW = 1.0 / MAX_YAW**2
W_YAW_TRACKING = 0.20  # Reward V2 value used by the selected TD3 baseline.


class CommandEnv(SesameRLEnv):
    def __init__(self, render_mode=None, stop_probability=STOP_PROBABILITY):
        self.command_vx = 0.0
        self.command_yaw = 0.0
        self._fixed_command = None
        self.stop_probability = float(stop_probability)
        if not 0.0 <= self.stop_probability <= 1.0:
            raise ValueError("stop_probability must be in [0, 1]")

        super().__init__(render_mode=render_mode, w_yaw=W_YAW_TRACKING)
        self.obs_dim += 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def set_command(self, vx, yaw):
        vx, yaw = float(vx), float(yaw)
        if not 0.0 <= vx <= MAX_VX:
            raise ValueError(f"vx must be in [0, {MAX_VX}] m/s")
        if not -MAX_YAW <= yaw <= MAX_YAW:
            raise ValueError(f"yaw must be in [-{MAX_YAW}, {MAX_YAW}] rad/s")
        self._fixed_command = (vx, yaw)
        self.command_vx, self.command_yaw = self._fixed_command

    def _sample_command(self):
        if self.np_random.random() < self.stop_probability:
            return 0.0, 0.0
        return (
            float(self.np_random.uniform(0.0, MAX_VX)),
            float(self.np_random.uniform(-MAX_YAW, MAX_YAW)),
        )

    def _get_obs(self):
        state = super()._get_obs()
        command = np.array(
            [self.command_vx / MAX_VX, self.command_yaw / MAX_YAW],
            dtype=np.float32,
        )
        return np.concatenate((state, command)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        if self._fixed_command is None:
            self.command_vx, self.command_yaw = self._sample_command()
        else:
            self.command_vx, self.command_yaw = self._fixed_command
        info.update(command_vx=self.command_vx, command_yaw=self.command_yaw)
        return self._get_obs(), info

    def _compute_reward(self, action):
        linear_velocity, angular_velocity = self._get_body_velocity()
        forward_velocity = float(linear_velocity[0])
        lateral_velocity = float(linear_velocity[1])
        yaw_rate = float(angular_velocity[2])
        upright = self._get_upright()
        forward_error = abs(forward_velocity - self.command_vx)
        yaw_error = abs(yaw_rate - self.command_yaw)

        reward_forward_tracking = W_FORWARD * np.exp(-K_V * forward_error**2)
        reward_yaw_tracking = self.w_yaw * np.exp(-K_YAW * yaw_error**2)
        penalty_lateral = W_LATERAL * abs(lateral_velocity)
        reward_upright = W_UPRIGHT * upright
        penalty_action_rate = W_ACTION_RATE * float(
            np.mean((action - self.previous_action) ** 2)
        )
        reward = (
            ALIVE_REWARD
            + reward_forward_tracking
            + reward_yaw_tracking
            - penalty_lateral
            + reward_upright
            - penalty_action_rate
        )

        info = {
            "command_vx": self.command_vx,
            "command_yaw": self.command_yaw,
            "forward_velocity": forward_velocity,
            "yaw_rate": yaw_rate,
            "lateral_velocity": lateral_velocity,
            "forward_error": forward_error,
            "yaw_error": yaw_error,
            "reward_forward_tracking": float(reward_forward_tracking),
            "reward_yaw_tracking": float(reward_yaw_tracking),
            "penalty_lateral": float(penalty_lateral),
            "reward_upright": float(reward_upright),
            "penalty_action_rate": float(penalty_action_rate),
            "alive_reward": float(ALIVE_REWARD),
            "upright": float(upright),
            "fall_penalty": float(FALL_PENALTY),
        }
        return float(reward), info


SesameCommandEnv = CommandEnv


def smoke_test():
    from stable_baselines3.common.env_checker import check_env

    env = CommandEnv()
    check_env(env, warn=True)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (35,) and env.action_space.shape == (8,)

    samples = []
    for seed in range(100):
        _, info = env.reset(seed=seed)
        samples.append((info["command_vx"], info["command_yaw"]))
    assert any(command == (0.0, 0.0) for command in samples)
    assert all(0.0 <= vx <= MAX_VX and -MAX_YAW <= yaw <= MAX_YAW for vx, yaw in samples)

    for command in [(0.0, 0.0), (0.20, 0.0), (0.0, 0.70), (0.0, -0.70)]:
        env.set_command(*command)
        obs, info = env.reset(seed=42)
        assert (info["command_vx"], info["command_yaw"]) == command
        assert np.allclose(obs[-2:], (command[0] / MAX_VX, command[1] / MAX_YAW))

    finite_keys = [
        "forward_velocity", "yaw_rate", "lateral_velocity", "forward_error",
        "yaw_error", "reward_forward_tracking", "reward_yaw_tracking", "upright",
    ]
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert all(np.isfinite(info[key]) for key in finite_keys)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    print("G1 PASS: observation=(35,), action=(8,), sampling/set_command/100-step rollout finite")


if __name__ == "__main__":
    smoke_test()
