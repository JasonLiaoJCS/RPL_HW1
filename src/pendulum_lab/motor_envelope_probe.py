"""Feedback-driven acceleration reversals and physical motor-envelope checks."""
import math
import json
import time
from pathlib import Path

import numpy as np

from .contracts import PROFILE
from .sysid_driven import DrivenSampler, check_settings, powered_session


REST_VELOCITY_RAD_S = .15
ZERO_POSITION_TOL_RAD = math.radians(1)


def _csv_floats(value):
    result = [float(part) for part in value.split(",")]
    if not result or not all(math.isfinite(v) and v > 0 for v in result):
        raise ValueError("Provide positive finite comma-separated values")
    return result


def _sample_until(sampler, *, accel, phase, deadline, done):
    while time.monotonic() < deadline:
        state = sampler.sample(accel, phase)
        if done(state):
            return state
    raise TimeoutError(f"Motion phase {phase} did not converge before timeout")


def _return_to_zero(sampler):
    # Refresh bounded acceleration control instead of sleeping in position mode.
    deadline = time.monotonic()+3.
    settled = 0
    while time.monotonic() < deadline:
        state = sampler.sample(lambda s: float(np.clip(
            -20*s.motor_pos_rad-10*s.motor_vel_rad_s, -40., 40.)), -1)
        if (abs(state.motor_pos_rad) <= ZERO_POSITION_TOL_RAD
                and abs(state.motor_vel_rad_s) <= REST_VELOCITY_RAD_S):
            settled += 1
            if settled*sampler.period >= .3:
                return
        else:
            settled = 0
    raise TimeoutError("Return-to-center did not settle")


def run_level(sampler, *, speed, accel, cycles):
    # Includes acceleration AND braking; firmware/transport lag needs margin too.
    excursion = speed*speed/accel + speed*.06
    if excursion > math.radians(50):
        raise ValueError("Requested speed/acceleration leaves insufficient travel margin")
    for cycle in range(cycles):
        direction = 1. if cycle % 2 == 0 else -1.
        _sample_until(sampler, accel=direction*accel, phase=2*cycle,
                      deadline=time.monotonic()+speed/accel+.20,
                      done=lambda s, d=direction: d*s.motor_vel_rad_s >= speed)
        _sample_until(sampler, accel=-direction*accel, phase=2*cycle+1,
                      deadline=time.monotonic()+speed/accel+.25,
                      done=lambda s, d=direction: d*s.motor_vel_rad_s <= REST_VELOCITY_RAD_S)
        _return_to_zero(sampler)


def configure_parser(parser):
    parser.add_argument("--port", required=True)
    parser.add_argument("--speeds", type=_csv_floats, default=[1.5, 2.5, 3.5])
    parser.add_argument("--accels", type=_csv_floats, default=[25., 50.])
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--sample-hz", type=float, default=200.)
    parser.add_argument("--out-dir", required=True, type=Path)


def execute(args):
    check_settings(args.sample_hz)
    if (not 2 <= args.cycles <= 10 or max(args.accels) > PROFILE.max_accel
            or max(args.speeds) > 5.):
        raise ValueError("Use 2-10 cycles, acceleration <=50 rad/s^2, speed <=5 rad/s")
    for speed in args.speeds:
        for accel in args.accels:
            if speed*speed/accel+speed*.06 > math.radians(50):
                raise ValueError("Requested level leaves insufficient travel margin")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    results = []
    summary = {"profile": "drv8825_32", "microsteps": PROFILE.microsteps,
               "baud": PROFILE.baud, "results": results, "complete": False}
    try:
        for speed in args.speeds:
            for accel in args.accels:
                stem = f"v{speed:g}_a{accel:g}".replace(".", "p")
                path = args.out_dir/f"{stem}.npz"
                if path.exists():
                    raise FileExistsError(path)
                print(f"Level speed={speed:g} rad/s, accel={accel:g} rad/s^2")
                # Each level has its own power-off preflight and explicit consent.
                with powered_session(args.port) as client:
                    sampler = DrivenSampler(client, args.sample_hz)
                    complete = False
                    try:
                        run_level(sampler, speed=speed, accel=accel, cycles=args.cycles)
                        complete = True
                    finally:
                        client.disengage_motor()
                        sampler.save(path, complete=complete, speed_rad_s=speed,
                                     accel_rad_s2=accel, cycles=args.cycles)
                passed = input("Did the PHYSICAL arm move, return within 2 degrees, "
                               "without grinding/skipping? Type YES: ").strip() == "YES"
                rows = np.asarray(sampler.rows)
                results.append({"speed_rad_s": speed, "accel_rad_s2": accel,
                                "cycles": args.cycles, "trace": path.name, "passed": passed,
                                "measured_peak_speed_rad_s": float(np.max(np.abs(rows[:, 5]))),
                                "measured_max_abs_position_rad": float(np.max(np.abs(rows[:, 3])))})
                if not passed:
                    raise RuntimeError("Physical motor check failed; ladder stopped")
        summary["complete"] = True
    finally:
        passed = [r for r in results if r["passed"]]
        point = passed[-1] if passed else None
        summary["recommendation_source"] = point
        for key in ("speed_rad_s", "accel_rad_s2"):
            summary[f"validated_{key}"] = point[key] if point else None
            summary[f"recommended_{key}"] = .8*point[key] if point else None
        (args.out_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return 0
