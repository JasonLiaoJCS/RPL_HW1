"""Supplied attended deployment loop. Students do not need serial access."""
from collections import deque
import math
import time
import numpy as np

from .contracts import PROFILE, checked_action, physical_state, observation_frame, wrap
from .serial_client import LowLevelClient


def validate_state(s, previous=None, startup=False):
    x = physical_state(s)
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite feedback")
    if abs(x[3]) > 50:
        raise RuntimeError("Pendulum speed/feedback guard")
    if previous is not None:
        delta_us = (s.time_us - previous) % 2**32
        if not 0 < delta_us < 100000:
            raise RuntimeError(f"Stale or discontinuous feedback (delta_us={delta_us})")
    if abs(x[0]) >= math.radians(PROFILE.arm_limit_deg) or abs(x[2]) > PROFILE.arm_speed_limit:
        raise RuntimeError("Arm travel/speed guard")
    if startup and (abs(x[0]) > math.radians(5) or abs(wrap(s.pendulum_pos_rad)) > math.radians(5)
                    or abs(x[2]) > 0.2 or abs(x[3]) > 0.5):
        raise RuntimeError("Start requires a physically centered arm and still hanging pendulum")
    return x


def preflight(client):
    if not client.wait_until_ready():
        raise RuntimeError("Firmware READY handshake failed")
    client.require_profile(microsteps=32, baud=500000, protocol_version=3)
    faults = client.get_safety_faults()
    if faults:
        raise RuntimeError(f"Firmware safety fault latched (bits={faults}); inspect before resetting")
    previous = None
    for _ in range(25):
        s = client.get_state()
        validate_state(s, previous, startup=True)
        previous = s.time_us
        time.sleep(PROFILE.dt)
    return s


def inspect(port):
    with LowLevelClient(port, timeout=0.1) as client:
        s = preflight(client)
        print(f"Preflight passed: arm={math.degrees(s.motor_pos_rad):.2f} deg, "
              f"hanging angle={math.degrees(s.pendulum_pos_rad):.2f} deg")


def run(controller, port, duration, rows, *, client_factory=LowLevelClient, confirm=input):
    if not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError("Hardware duration must be in (0, 30]")
    # Fail on unimplemented controllers before opening USB or enabling outputs.
    test_obs = np.tile(observation_frame([0, -math.pi, 0, 0], 0), PROFILE.history)
    controller.reset()
    checked_action(controller.act(test_obs))
    controller.reset()
    with client_factory(port, timeout=0.1) as client:
        preflight(client)
        if confirm("Preflight passed. Power ON, keep clear, type RUN: ").strip() != "RUN":
            raise RuntimeError("Powered run cancelled")
        # Keep this same USB connection: reopening a Nano can reset and tare it.
        s = client.get_state()
        validate_state(s, startup=True)
        history = deque([observation_frame(physical_state(s), 0)] * PROFILE.history,
                        maxlen=PROFILE.history)
        previous = s.time_us
        last_action = 0.0
        client.engage_motor()
        try:
            start = time.monotonic()
            for tick in range(round(duration * PROFILE.hz)):
                # GET_STATE returns a cached 500 Hz sample. Allow a new sample
                # after the startup read instead of requesting tick zero immediately.
                deadline = start + (tick + 1) * PROFILE.dt
                time.sleep(max(0, deadline - time.monotonic()))
                s = client.get_state()
                x = validate_state(s, previous)
                previous = s.time_us
                if time.monotonic() - deadline > 0.04:
                    raise RuntimeError("Feedback deadline missed")
                history.append(observation_frame(x, last_action))
                action = checked_action(controller.act(np.concatenate(history)))
                if time.monotonic() - deadline > 0.04:
                    raise RuntimeError("Controller deadline missed")
                # A watchdog trip requires a new attended run, never re-engage.
                if not client.get_motor_status().motor_engaged:
                    raise RuntimeError("Firmware motor disabled or watchdog tripped")
                if time.monotonic() - deadline > 0.04:
                    raise RuntimeError("Command deadline missed")
                client.set_acceleration(action * PROFILE.max_accel)
                last_action = action
                rows.append([time.monotonic() - start, *x, action])
        finally:
            client.disengage_motor()


def motor_check(port):
    """One fixed 533-microstep move, approximately +30 degrees at 1/32."""
    with LowLevelClient(port, timeout=0.1) as client:
        preflight(client)
        if input("Power ON, keep clear, type MOVE for a +30 degree check: ").strip() != "MOVE":
            raise RuntimeError("Motor check cancelled")
        validate_state(client.get_state(), startup=True)
        client.engage_motor()
        try:
            client.start_motor_calibration()
            start = time.monotonic()
            previous = None
            while time.monotonic() - start < 5:
                s = client.get_state()
                validate_state(s, previous)
                previous = s.time_us
                if abs(s.motor_pos_rad) > math.radians(35):
                    raise RuntimeError("Calibration travel guard")
                status = client.get_motor_status()
                if not status.planner_running:
                    error = abs(s.motor_pos_rad - math.radians(30))
                    if error > math.radians(2):
                        raise RuntimeError("Calibration stopped before target")
                    print("Reported target reached; verify PHYSICAL travel yourself.")
                    return
                time.sleep(PROFILE.dt)
            raise RuntimeError("Calibration timed out")
        finally:
            client.disengage_motor()
