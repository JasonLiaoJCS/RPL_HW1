"""Shared guarded acquisition for attended driven-identification experiments."""
from contextlib import contextmanager
import math
import time

import numpy as np

from .contracts import PROFILE
from .hardware import preflight, validate_state
from .serial_client import LowLevelClient


COLUMNS = np.asarray([
    "host_time_s", "phase", "accel_cmd_rad_s2", "motor_pos_rad",
    "pendulum_pos_rad", "motor_vel_rad_s", "pendulum_vel_rad_s",
])


def check_settings(sample_hz):
    if not math.isfinite(sample_hz) or not 50 <= sample_hz <= 200:
        raise ValueError("Sampling must be between 50 and 200 Hz")


@contextmanager
def powered_session(port, *, confirm=input, client_factory=LowLevelClient):
    if confirm("12 V OFF, arm centered, pendulum hanging still? Type OFF: ").strip() != "OFF":
        raise RuntimeError("Power-off confirmation required")
    try:
        with client_factory(port, baud=PROFILE.baud, timeout=0.1) as client:
            client.disengage_motor()
            preflight(client)
            if confirm("Power ON, clear the swing area, keep switch accessible. Type RUN: ").strip() != "RUN":
                raise RuntimeError("Powered experiment cancelled")
            validate_state(client.get_state(), startup=True)
            if client.get_safety_faults():
                raise RuntimeError("Firmware safety fault")
            try:
                client.set_acceleration(0.0)
                client.engage_motor()
                yield client
            finally:
                client.disengage_motor()
    finally:
        print("Switch 12 V OFF before touching or recentering the mechanism.")


class DrivenSampler:
    """50 Hz commands, faster feedback, continuous checks across motion phases."""

    def __init__(self, client, sample_hz, *, position_limit_deg=60.0):
        check_settings(sample_hz)
        self.client = client
        self.period = 1.0 / sample_hz
        self.limit = math.radians(min(position_limit_deg, PROFILE.arm_limit_deg))
        self.rows = []
        self.device_time_us = []
        self.command_times = []
        self.commands = []
        self.previous = None
        self.start = time.monotonic()
        self.next_sample = self.start + self.period
        self.next_command = self.start
        self.last_accel = 0.0
        self.last_phase = -1

    def sample(self, command, phase):
        time.sleep(max(0.0, self.next_sample - time.monotonic()))
        deadline = self.next_sample
        self.next_sample += self.period
        if time.monotonic() - deadline > 0.04:
            raise RuntimeError("Feedback deadline missed")
        if time.monotonic() - self.start > 15:
            raise RuntimeError("Driven experiment exceeded 15 seconds")
        s = self.client.get_state()
        stamp = time.monotonic()
        validate_state(s, self.previous)
        self.previous = s.time_us
        if abs(s.motor_pos_rad) >= self.limit:
            raise RuntimeError("Driven experiment position guard")
        self.rows.append((stamp, self.last_phase, self.last_accel, s.motor_pos_rad,
                          s.pendulum_pos_rad, s.motor_vel_rad_s, s.pendulum_vel_rad_s))
        self.device_time_us.append(s.time_us)
        # Check the latch/status before every possible output, never re-engage.
        if stamp >= self.next_command:
            if self.client.get_safety_faults():
                raise RuntimeError("Firmware safety fault")
            if not self.client.get_motor_status().motor_engaged:
                raise RuntimeError("Motor disabled or watchdog tripped")
            accel = float(command(s) if callable(command) else command)
            if not math.isfinite(accel) or abs(accel) > PROFILE.max_accel:
                raise ValueError("Acceleration exceeds the lab cap")
            now = time.monotonic()
            if now - deadline > 0.04:
                raise RuntimeError("Command deadline missed")
            self.client.set_acceleration(accel)
            self.command_times.append(time.monotonic() - self.start)
            self.commands.append(accel)
            self.last_accel = accel
            self.last_phase = phase
            self.next_command += PROFILE.dt
        return s

    def save(self, path, *, complete, **metadata):
        np.savez_compressed(
            path, samples=np.asarray(self.rows, dtype=float).reshape(-1, 7),
            columns=COLUMNS, device_time_us=np.asarray(self.device_time_us, dtype=np.uint32),
            command_t=np.asarray(self.command_times), accel_commands=np.asarray(self.commands),
            start_host_time_s=self.start, complete=complete,
            command_hz=PROFILE.hz, **metadata,
        )
