"""One state, observation and action convention for all seven algorithms."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

PARAMS = Path(__file__).parent / "plant" / "data" / "example_rig.json"


@dataclass(frozen=True)
class Profile:
    hz: int = 50
    max_accel: float = 50.0
    arm_limit_deg: float = 100.0
    arm_speed_limit: float = 6.0
    history: int = 4
    microsteps: int = 32
    baud: int = 500000

    @property
    def dt(self):
        return 1.0 / self.hz

    @property
    def obs_dim(self):
        return 6 * self.history


PROFILE = Profile()


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def physical_state(serial_state):
    return np.array([serial_state.motor_pos_rad,
                     wrap(serial_state.pendulum_pos_rad - math.pi),
                     serial_state.motor_vel_rad_s, serial_state.pendulum_vel_rad_s])


def state_from_obs(obs):
    frame = np.asarray(obs, dtype=float)[-6:]
    if frame.shape != (6,) or not np.isfinite(frame).all():
        raise ValueError("Expected finite six-value frames")
    return np.array([frame[0], math.atan2(frame[1], frame[2]), frame[3], frame[4]])


def observation_frame(state, previous_action):
    arm, theta, arm_vel, pen_vel = state
    return np.array([arm, math.sin(theta), math.cos(theta), arm_vel,
                     pen_vel, previous_action], dtype=np.float32)


def checked_action(value):
    a = np.asarray(value)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError("Controller must return one finite normalized acceleration")
    return float(np.clip(a.item(), -1.0, 1.0))


def manifest(params):
    return {"profile": asdict(PROFILE), "action_mode": "accel",
            "params_sha256": hashlib.sha256(Path(params).read_bytes()).hexdigest()}
