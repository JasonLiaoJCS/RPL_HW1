"""Canonical assignment environment, with the hardware envelope in simulation."""
import math
import gymnasium as gym
import numpy as np

from .contracts import PROFILE
from .plant.pendulum_env import RotaryInvertedPendulumEnv


class LabEnvelope(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pen = self.unwrapped
        arm_vel = float(pen.data.qvel[pen._motor_qvel_addr])
        if abs(info["motor_pos"]) >= math.radians(PROFILE.arm_limit_deg) or abs(arm_vel) > PROFILE.arm_speed_limit:
            info["lab_guard"] = True
            terminated = True
            reward -= 100.0
        return obs, reward, terminated, truncated, info


def make_env(params, *, duration=15.0, start="hanging", render=False):
    if start not in ("hanging", "upright", "mixed"):
        raise ValueError("Unknown reset distribution")
    if not math.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError("Simulation duration must be in (0, 60]")
    return LabEnvelope(RotaryInvertedPendulumEnv(
        params_path=params, action_mode="accel", control_freq_hz=PROFILE.hz,
        max_accel_rad_s2=PROFILE.max_accel, action_smooth_window=1,
        obs_history_len=PROFILE.history, firmware_obs_model=True,
        episode_length_s=duration,
        upright_reset_frac=1.0 if start == "upright" else (0.5 if start == "mixed" else 0.0),
        exact_hanging_reset_frac=1.0 if start == "hanging" else (0.5 if start == "mixed" else 0.0),
        transport_delay_s=0.010, motor_planner_speed_limit_rad_s=5.0,
        reward_alive_offset=15, reward_upright_alive_weight=5,
        reward_stillness_bonus_weight=5, reward_upright_overspeed_weight=0.02,
        terminate_on_spin_loop=False, render_mode="human" if render else None))


def true_state(env):
    from .contracts import wrap
    e = env.unwrapped
    return np.array([e.data.qpos[e._motor_qpos_addr],
                     wrap(e.data.qpos[e._pen_qpos_addr] - math.pi),
                     e.data.qvel[e._motor_qvel_addr], e.data.qvel[e._pen_qvel_addr]])
