"""Interactive system-identification wizard.

Uses the same firmware as deployment. Free-swing identification assumes a
fixed arm; students must validate whether the model describes their rig.

The wizard splits cleanly into two phases:

    collect    — operator-driven data acquisition. Writes raw recordings
                 (NumPy .npz + a metadata.json) to a timestamped directory.
                 Includes an attended powered sweep unless --skip-motor is set.
    fit        — pure post-processing. Reads a recordings directory,
                 derives parameters, generates sim-vs-real validation
                 plots, and writes the per-rig sysid_params_*.json. No device required.

Recorded data can be re-fitted without reconnecting the hardware.
Use a separate `--out-json` path for each rig's fitted parameters.

Usage:
    # full pipeline (collect+fit+validate)
    pendulum-lab sysid --out-json configs/my_rig.json
    pendulum-lab sysid collect                  # only record data
    pendulum-lab sysid fit --in-dir <path> --out-json configs/my_rig.json
    pendulum-lab sysid validate-motor           # only the motor ±90° sanity sweep
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .serial_client import LowLevelClient
from .contracts import PROFILE
from .hardware import preflight, validate_state

MOTOR_MICROSTEPS = PROFILE.microsteps
from .sysid_core import (
    DEFAULT_FREE_SWING_DURATION_S,
    DEFAULT_FREE_SWING_RUNS,
    aggregate_free_swing_fits,
    derive_pendulum_friction,
    fit_free_swing,
    suggest_control_rate,
    validate_free_swing,
    write_params_json,
)


# ---------------------------------------------------------------------------
# Terminal helpers (ANSI; degrade gracefully when stdout isn't a tty)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(s: str) -> str: return _c("1", s)
def _dim(s: str) -> str: return _c("2", s)
def _yellow(s: str) -> str: return _c("33", s)
def _red(s: str) -> str: return _c("31", s)
def _green(s: str) -> str: return _c("32", s)
def _cyan(s: str) -> str: return _c("36", s)


def _hr(title: str = "") -> None:
    width = 72
    if title:
        print("\n" + _bold(f"── {title} ".ljust(width, "─")))
    else:
        print("\n" + ("─" * width))


def _print_warnings(warnings) -> None:
    if not warnings:
        return
    for w in warnings:
        if w.severity == "error":
            print(_red(f"  [error] {w.message}"))
        elif w.severity == "warn":
            print(_yellow(f"  [warn]  {w.message}"))
        else:
            print(_dim(f"  [info]  {w.message}"))


def _prompt_yes_no(label: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        s = input(f"{label} {suffix}: ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def _prompt_enter(label: str) -> None:
    input(f"{label} (press Enter)")


def _countdown(seconds: int = 3) -> None:
    for k in range(seconds, 0, -1):
        print(f"    {k}...")
        time.sleep(1)
    print(_bold("    GO."))


# ---------------------------------------------------------------------------
# USB serial port discovery
# ---------------------------------------------------------------------------

def auto_detect_port() -> list[str]:
    """Return candidate Arduino-compatible USB serial ports."""
    import serial.tools.list_ports
    needles = ("usbserial", "usbmodem", "wchusbserial", "ttyUSB", "ttyACM")
    return [
        p.device for p in serial.tools.list_ports.comports()
        if any(n in p.device for n in needles)
    ]


def pick_port(arg_port: str | None) -> str:
    if arg_port:
        return arg_port
    candidates = auto_detect_port()
    if not candidates:
        print(_red("No USB-serial ports detected. Pass --port explicitly."))
        sys.exit(1)
    if len(candidates) == 1:
        return candidates[0]
    print(_bold("Multiple candidate ports found:"))
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p}")
    while True:
        s = input("Pick one [0]: ").strip() or "0"
        try:
            idx = int(s)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
        print(_red("  invalid choice, try again"))


def make_output_dir(arg_out_dir: str | None) -> Path:
    if arg_out_dir:
        out = Path(arg_out_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = Path("runs") / "sysid" / ts
    out.mkdir(parents=True, exist_ok=False)
    return out


# ---------------------------------------------------------------------------
# Recording helpers — use LowLevelClient with motor disengaged (free-swing)
# or engaged with bounded PD control (motor sweep).
# ---------------------------------------------------------------------------

def _record_state_stream(
    client: LowLevelClient, duration_s: float, sample_hz: float
) -> dict:
    """Sample state at `sample_hz` for `duration_s`. Returns a dict of arrays.

    Values arrive in the firmware frame — the single frame used everywhere.
    """
    if (not np.isfinite([duration_s, sample_hz]).all() or not 0 < duration_s <= 60
            or not 20 <= sample_hz <= 100):
        raise ValueError("Invalid bounded recording duration/rate")
    n = int(duration_s * sample_hz)
    dt = 1.0 / sample_hz
    t = np.zeros(n, dtype=np.float64)
    pen = np.zeros(n, dtype=np.float64)
    motor = np.zeros(n, dtype=np.float64)
    pvel = np.zeros(n, dtype=np.float64)
    mvel = np.zeros(n, dtype=np.float64)

    t_start = time.monotonic()
    next_tick = t_start
    previous = None
    for i in range(n):
        next_tick += dt
        time.sleep(max(0, next_tick - time.monotonic()))
        s = client.get_state()
        validate_state(s, previous)
        previous = s.time_us
        if client.get_safety_faults():
            raise RuntimeError("Firmware safety fault; inspect the rig before resetting")
        if client.get_motor_status().motor_engaged:
            raise RuntimeError("Free-swing recording requires motor outputs disabled")
        if time.monotonic() - next_tick > .04:
            raise RuntimeError("Recording deadline missed")
        t[i] = time.monotonic() - t_start
        pen[i] = s.pendulum_pos_rad
        motor[i] = s.motor_pos_rad
        pvel[i] = s.pendulum_vel_rad_s
        mvel[i] = s.motor_vel_rad_s

    return dict(t=t, pen=pen, motor=motor, pvel=pvel, mvel=mvel)


def record_tare(client: LowLevelClient, out_dir: Path,
                duration_s: float = 3.0, sample_hz: float = 100.0) -> float:
    """Sample a few seconds of pendulum-at-rest to establish the hanging zero.

    The firmware's pendulum-angle accumulator has an arbitrary offset (the
    raw AS5600 reading at Arduino boot is its zero). Recording while
    everything is at rest lets us pin "physical hanging" to a known value
    in the saved coordinate frame, then express every subsequent recording
    relative to it. Removes the operator-error class where the user
    starts the wizard with the pendulum at an unknown angle.
    """
    _hr("Tare hanging position")
    print(_dim(
        "  Let the pendulum hang at rest (motor disengaged). Don't touch\n"
        "  it during the recording. We sample for a few seconds and use\n"
        "  the median as 'physical hanging = 0' for all downstream fits."
    ))
    client.disengage_motor()
    time.sleep(0.5)
    _prompt_enter("\n  Pendulum still and undisturbed?")

    print(_dim(f"  Recording {duration_s:.0f}s..."))
    rec = _record_state_stream(client, duration_s, sample_hz)
    hanging_zero = float(np.median(rec["pen"]))
    motion = float(np.std(rec["pen"]))
    print(_dim(f"  Hanging zero: {hanging_zero:+.4f} rad   "
               f"(std during tare: {motion*1000:.2f} mrad)"))
    if motion > 0.01:
        raise RuntimeError("Pendulum moved during tare (>10 mrad std); repeat with it still")
    np.savez(out_dir / "tare.npz", **rec, hanging_zero=hanging_zero)
    return hanging_zero


def record_free_swing(
    client: LowLevelClient, out_dir: Path, run_i: int, n_runs: int,
    duration_s: float, sample_hz: float = 100.0,
) -> Path:
    """Record one free-swing trial. Motor is disengaged; operator releases."""
    print(_bold(f"\n  Run {run_i + 1}/{n_runs}"))
    print(_dim(
        "    Hold the motor arm firmly. Slide a box under the pendulum so\n"
        "    its tip rests on the box (the lift angle doesn't have to be\n"
        "    exact — the sensor will record whatever angle you start from).\n"
        "    When ready, press Enter. After the countdown, slide the box\n"
        "    out perpendicular to the swing plane to release."
    ))
    _prompt_enter("    Ready to release")
    print("    Releasing in...")
    _countdown(3)
    print(_dim(f"    Recording for {duration_s:.0f}s..."))
    rec = _record_state_stream(client, duration_s, sample_hz)

    # Quick sanity: did the pendulum actually move?
    amp = float(np.max(np.abs(rec["pen"] - np.mean(rec["pen"]))))
    if amp < 0.05:
        print(_yellow(f"    WARNING: only {amp:.3f} rad of pendulum motion "
                      "— did the release work?"))
    else:
        print(_dim(f"    -> max amplitude during recording: {amp:.3f} rad "
                   f"({math.degrees(amp):.0f}°)"))

    path = out_dir / f"free_run_{run_i + 1}.npz"
    np.savez(path, **rec)
    return path


def move_motor_to(
    client: LowLevelClient, target_pos: float, *,
    max_accel: float = 40.0, kp: float = 20.0, kd: float = 10.0,
    tolerance_rad: float = 0.03, control_hz: float = 50.0,
    timeout_s: float = 4.0, log: list | None = None,
    log_t_offset: float = 0.0,
) -> float:
    """Drive the motor to `target_pos` with a closed-loop PD controller
    on acceleration.

    Acceleration is bounded and feedback is checked throughout settling.
    Motor position comes from step counts, not an independent arm encoder.
    """
    if (not np.isfinite([target_pos, max_accel, kp, kd, tolerance_rad, control_hz, timeout_s]).all()
            or abs(target_pos) > math.pi / 2 or not 0 < max_accel <= PROFILE.max_accel
            or control_hz != PROFILE.hz or not 0 < timeout_s <= 5
            or min(kp, kd, tolerance_rad) <= 0):
        raise ValueError("Invalid bounded motor-sweep settings")
    dt = 1.0 / control_hz
    next_tick = time.monotonic()
    t_start = next_tick
    settle_count = 0
    # Continue feedback and watchdog updates throughout the 0.7 s settling hold.
    settle_needed = int(0.7 * control_hz)
    previous = None
    while time.monotonic() - t_start < timeout_s:
        next_tick += dt
        time.sleep(max(0, next_tick - time.monotonic()))
        s = client.get_state()
        validate_state(s, previous)
        previous = s.time_us
        if client.get_safety_faults() or not client.get_motor_status().motor_engaged:
            raise RuntimeError("Motor disabled or firmware fault; sweep aborted")
        if time.monotonic() - next_tick > .04:
            raise RuntimeError("Motor command deadline missed")
        motor_pos = s.motor_pos_rad
        motor_vel = s.motor_vel_rad_s

        error = target_pos - motor_pos
        accel = kp * error - kd * motor_vel
        accel = float(np.clip(accel, -max_accel, max_accel))

        client.set_acceleration(accel)

        if log is not None:
            log.append((time.monotonic() - t_start + log_t_offset,
                        motor_pos, motor_vel, accel, target_pos))

        if abs(error) < tolerance_rad and abs(motor_vel) < 0.3:
            settle_count += 1
            if settle_count >= settle_needed:
                break
        else:
            settle_count = 0

    if settle_count < settle_needed:
        raise RuntimeError("Motor segment did not settle before timeout")
    client.set_acceleration(0.0)
    return time.monotonic() - t_start


def record_motor_sweep(client: LowLevelClient, out_dir: Path) -> Path:
    """Attended 0, +90, 0, -90, 0 degree sweep with protocol-3 guards."""
    _hr("Motor +/-90 degree sanity sweep")
    print("Step counts cannot establish physical travel or detect missed steps.")
    if input("Switch 12 V OFF, remove the arm support, center physically, "
             "and let the pendulum hang still. Type OFF: ").strip() != "OFF":
        raise RuntimeError("Motor sweep cancelled")
    client.disengage_motor()
    preflight(client)
    if input("Power ON, clear the full swing area, keep the switch accessible. "
             "Type SWEEP: ").strip() != "SWEEP":
        raise RuntimeError("Motor sweep cancelled")
    validate_state(client.get_state(), startup=True)
    if client.get_safety_faults():
        raise RuntimeError("Firmware fault before engagement")
    log = []
    path = out_dir / "motor_sweep.npz"
    save_metadata(out_dir, motor_sweep_state="running", physical_travel_confirmed=False)
    start = time.monotonic()
    try:
        client.engage_motor()
        for target in (0., math.pi / 2, 0., -math.pi / 2, 0.):
            print(f"  Target {math.degrees(target):+.0f} degrees")
            move_motor_to(client, target, log=log, log_t_offset=time.monotonic() - start)
        save_metadata(out_dir, motor_sweep_state="complete")
    except BaseException as exc:
        save_metadata(out_dir, motor_sweep_state="failed", motor_sweep_error=str(exc))
        raise
    finally:
        try:
            client.disengage_motor()
        finally:
            rows = np.asarray(log, dtype=float).reshape(-1, 5)
            np.savez(path, t=rows[:, 0], motor=rows[:, 1], mvel=rows[:, 2],
                     accel_cmd=rows[:, 3], target=rows[:, 4])
            print("Sweep ended. Switch 12 V OFF before touching or recentering.")
    confirmed = input("Did the PHYSICAL arm reach both +/-90 degree marks? Type YES: ").strip() == "YES"
    save_metadata(out_dir, physical_travel_confirmed=confirmed)
    if not confirmed:
        raise RuntimeError("Physical motor travel unverified; do not deploy a policy")
    return path

# ---------------------------------------------------------------------------
# Metadata save/load
# ---------------------------------------------------------------------------

def save_metadata(out_dir: Path, **kwargs) -> None:
    meta_path = out_dir / "metadata.json"
    existing: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing.update(kwargs)
    existing.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)


def load_metadata(in_dir: Path) -> dict:
    with open(in_dir / "metadata.json") as f:
        return json.load(f)


def load_free_recording(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        rec = {key: np.asarray(data[key], dtype=float) for key in ("t", "pen", "motor", "pvel", "mvel")}
    length = len(rec["t"])
    if length < 30 or any(value.shape != (length,) or not np.isfinite(value).all() for value in rec.values()):
        raise ValueError("Invalid or too-short recording")
    if np.any(np.diff(rec["t"]) <= 0):
        raise ValueError("Recording timestamps must increase")
    if np.ptp(rec["motor"]) > math.radians(1) or np.max(np.abs(rec["mvel"])) > .1:
        raise ValueError("Free swing requires a fixed arm")
    return rec


# ---------------------------------------------------------------------------
# Collection phase — device-bound
# ---------------------------------------------------------------------------

def collect_data(args: argparse.Namespace) -> Path:
    port = pick_port(args.port)
    out_dir = make_output_dir(args.out_dir)
    save_metadata(out_dir, collection_state="running", firmware="LowLevelServer.ino",
                  port=port, baud=args.baud, profile=asdict(PROFILE),
                  n_free_runs=args.n_free_runs, free_swing_duration_s=args.duration_s,
                  sample_hz=args.sample_hz)
    print(f"Recording session: {out_dir}")
    print("Geometry comes from packaged CAD. Validate it for your assembled kit.")
    try:
        if input("Keep 12 V OFF, USB connected, arm centered and pendulum hanging. Type OFF: ").strip() != "OFF":
            raise RuntimeError("Collection cancelled")
        with LowLevelClient(port, baud=args.baud, timeout=.1) as client:
            client.disengage_motor()
            preflight(client)
            hanging_zero = record_tare(client, out_dir)
            save_metadata(out_dir, hanging_zero_rad=hanging_zero)
            print("Keep 12 V OFF. Hold or mechanically secure the arm during free swings.")
            for i in range(args.n_free_runs):
                record_free_swing(client, out_dir, i, args.n_free_runs,
                                  args.duration_s, args.sample_hz)
            save_metadata(out_dir, free_swing_state="complete")
            if not args.skip_motor:
                record_motor_sweep(client, out_dir)
        save_metadata(out_dir, collection_state="complete")
    except BaseException as exc:
        save_metadata(out_dir, collection_state="failed", collection_error=str(exc))
        raise
    print(f"Collection complete: {out_dir}")
    return out_dir

# ---------------------------------------------------------------------------
# Fit phase — pure post-processing
# ---------------------------------------------------------------------------

def fit_data(in_dir: Path, out_json: Path) -> dict:
    _hr("Fit + analysis")
    print(_dim(f"  In:  {in_dir}"))
    print(_dim(f"  Out: {out_json}\n"))

    if out_json.exists():
        raise FileExistsError(out_json)
    meta = load_metadata(in_dir)
    if meta.get("collection_state", "complete") != "complete":
        raise ValueError("Collection incomplete or failed; inspect the recordings before fitting")
    hanging_zero = float(meta.get("hanging_zero_rad", 0.0))

    free_files = sorted(in_dir.glob("free_run_*.npz"))
    if not free_files:
        print(_red("  No free_run_*.npz files in directory."))
        raise ValueError("No free-swing recordings")

    print(_dim(f"  {len(free_files)} free-swing recording(s)"))
    fits: list[dict] = []
    sources = []
    for path in free_files:
        rec = load_free_recording(path)
        pen_centered = rec["pen"] - hanging_zero
        try:
            fit = fit_free_swing(rec["t"], pen_centered)
        except Exception as e:
            print(_red(f"    {path.name}: fit failed ({e})"))
            continue
        fits.append(fit)
        sources.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(_dim(
            f"    {path.name}: T={fit['period_s']*1000:.0f} ms  "
            f"T₀={fit['small_amp_period_s']*1000:.0f} ms  "
            f"α={fit['decay_constant_s_inv']:.4f} 1/s  "
            f"peaks={fit['n_extrema_used']}"
        ))
    if not fits:
        print(_red("  All fits failed."))
        raise ValueError("All free-swing fits failed")

    fit = aggregate_free_swing_fits(fits)
    derived = derive_pendulum_friction(fit)

    rel_err = abs(
        derived["inertia_measured_kg_m2"] - derived["inertia_predicted_kg_m2"]
    ) / derived["inertia_predicted_kg_m2"]
    print(_bold("\n  Derived parameters:"))
    print(f"    viscous friction:        {derived['viscous_friction_N_m_s']:.4e} N·m·s")
    print(f"    Coulomb friction:        {derived['coulomb_friction_N_m']:.4e} N·m")
    print(_dim(
        f"    I (CAD-predicted):       {derived['inertia_predicted_kg_m2']:.4e} kg·m²"
    ))
    print(_dim(
        f"    I (measured from T₀):    {derived['inertia_measured_kg_m2']:.4e} kg·m²  "
        f"({rel_err*100:+.1f}% vs CAD)"
    ))

    # Validate
    warnings = validate_free_swing(fit, derived)
    if warnings:
        print()
        _print_warnings(warnings)

    # Diagnostic only; the lab control rate remains fixed at 50 Hz.
    print(_bold("\n  Control rate suggestion (based on pendulum dynamics alone):"))
    from .contracts import PROFILE
    CONTROL_FREQ_HZ = PROFILE.hz
    pendulum_period = fit["small_amp_period_s"]
    f_n = 1.0 / pendulum_period
    print(f"    pendulum natural freq f_n ≈ {f_n:.2f} Hz")
    print(f"    minimum useful control rate (5·f_n) ≈ {5*f_n:.1f} Hz")
    print(_dim(f"    (the rig deploys at {CONTROL_FREQ_HZ:.0f} Hz; this run won't change that)"))

    # Store fitted friction and period-derived inertia alongside CAD diagnostics.
    pendulum_payload = {
        "derived": {
            "viscous_friction_N_m_s": derived["viscous_friction_N_m_s"],
            "coulomb_friction_N_m": derived["coulomb_friction_N_m"],
            "inertia_predicted_kg_m2": derived["inertia_predicted_kg_m2"],
            "inertia_measured_kg_m2": derived["inertia_measured_kg_m2"],
        },
        "fit": fit,
    }
    run_meta = {
        "wizard_version": "2026-05-20",
        "collected_at": meta.get("timestamp"),
        "fitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "in_dir": str(in_dir.resolve()),
        "n_free_swing_runs": len(fits),
        "hanging_zero_rad": hanging_zero,
        "firmware": meta.get("firmware", "unknown"),
        "training_sources": sources,
        "method": "original_peak_envelope_and_CAD_friction",
    }
    from .plant.pendulum_geometry import PENDULUM_MASS_KG, PENDULUM_COM_M
    run_meta.update(pendulum_total_mass_kg=PENDULUM_MASS_KG, pendulum_com_m=PENDULUM_COM_M)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_params_json(str(out_json),
                      pendulum_payload=pendulum_payload, run_meta=run_meta)
    print(_green(f"\n  Wrote {out_json}"))

    # Generate validation plots
    plot_free_swing_compare(in_dir, derived, out_dir=in_dir, params_path=out_json)
    if (in_dir / "motor_sweep.npz").exists():
        plot_motor_sweep(in_dir / "motor_sweep.npz", out_dir=in_dir)

    return derived


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_free_swing_compare(in_dir: Path, derived: dict, *, out_dir: Path,
                            params_path: Path, recording: Path | None = None) -> dict:
    """Replay the longest free-swing in sim and compare envelopes.

    Time-aligned to the actual release moment, not the recording start —
    the operator typically holds the pendulum on the box for some
    fraction of a second after pressing GO. We detect release as the
    first sample whose pen-velocity exceeds a small threshold, then
    initialise sim from that sample's (pos, vel) state and plot both
    traces with that moment as t=0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    from .plant.pendulum_env import RotaryInvertedPendulumEnv

    free_files = sorted(in_dir.glob("free_run_*.npz"))
    if not free_files:
        return
    rec = load_free_recording(recording or free_files[0])
    meta = load_metadata(in_dir)
    hanging_zero = float(meta.get("hanging_zero_rad", 0.0))

    real_t = rec["t"]
    real_pen = rec["pen"] - hanging_zero
    real_pvel = rec["pvel"]
    sample_dt = float(np.median(np.diff(real_t)))

    # Find the release index: first sample where |pen_vel| > 0.5 rad/s
    # (well above sensor noise, well below typical release speeds).
    moving = np.where(np.abs(real_pvel) > 0.5)[0]
    rel_idx = int(moving[0]) if len(moving) else 0
    # Step back a sample so the trace shows the very start of motion.
    rel_idx = max(0, rel_idx - 1)

    # Trim both traces to start at release.
    real_t_trim = real_t[rel_idx:] - real_t[rel_idx]
    real_pen_trim = real_pen[rel_idx:]
    release_pos = float(real_pen_trim[0])
    release_vel = float(real_pvel[rel_idx])

    # Pass the fitted file explicitly; the simulator's default is only a starter.
    env = RotaryInvertedPendulumEnv(
        params_path=params_path, action_mode="accel", action_smooth_window=1,
        control_freq_hz=200.0,
        episode_length_s=real_t_trim[-1] + 0.5,
    )
    env.reset(seed=0)
    env.data.qpos[env._motor_qpos_addr] = 0.0
    env.data.qpos[env._pen_qpos_addr]   = release_pos
    env.data.qvel[env._motor_qvel_addr] = 0.0
    env.data.qvel[env._pen_qvel_addr]   = release_vel
    env._motor_target = 0.0; env._motor_vel = 0.0
    mujoco.mj_forward(env.model, env.data)

    n = len(real_t_trim)
    sim_pen = np.zeros(n)
    try:
        simulated_t = 0.
        for idx, target_t in enumerate(real_t_trim):
            while simulated_t + 1e-10 < target_t:
                env.step(np.array([0.0], dtype=np.float32))
                simulated_t += 1.0 / 200.0
            sim_pen[idx] = float(env.data.qpos[env._pen_qpos_addr])
    finally:
        env.close()

    # Use trimmed arrays for the rest of the function.
    real_t = real_t_trim
    real_pen = real_pen_trim

    # Fit envelopes
    def _fit(t, y):
        crossings = np.where(np.diff(np.sign(y)) != 0)[0]
        pt, pa = [], []
        for k in range(len(crossings) - 1):
            a, b = crossings[k] + 1, crossings[k + 1] + 1
            j = a + int(np.argmax(np.abs(y[a:b])))
            pt.append(t[j]); pa.append(abs(y[j]))
        pt, pa = np.array(pt), np.array(pa)
        if len(pt) < 3:
            return None
        period = 2 * np.median(np.diff(pt))
        decay = -float(np.polyfit(pt, np.log(pa + 1e-9), 1)[0])
        return period, decay, pt, pa

    ix_start = int(0.3 / sample_dt)
    fr = _fit(real_t[ix_start:], real_pen[ix_start:])
    fs = _fit(real_t[ix_start:], sim_pen[ix_start:])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(real_t, real_pen, label="real", lw=1.0)
    axes[0].plot(real_t, sim_pen, label="sim (from derived params)",
                  lw=1.0, alpha=0.85)
    axes[0].axhline(0, color="k", lw=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("angle from hanging (rad)")
    axes[0].grid(alpha=0.3)
    if fr and fs:
        axes[0].set_title(
            f"real T={fr[0]*1000:.0f} ms  τ={fr[1]:.3f}/s  vs  "
            f"sim T={fs[0]*1000:.0f} ms  τ={fs[1]:.3f}/s  "
            f"(Δ {(fs[0]/fr[0]-1)*100:+.1f}%, {(fs[1]/fr[1]-1)*100:+.1f}%)"
        )
        axes[1].semilogy(fr[2], fr[3], "o-", label=f"real peaks (τ={fr[1]:.3f}/s)")
        axes[1].semilogy(fs[2], fs[3], "s--", label=f"sim peaks (τ={fs[1]:.3f}/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("|peak amplitude| (rad, log)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = out_dir / "freeswing_compare.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(_dim(f"  -> {fig_path.name}"))
    np.savetxt(out_dir / "freeswing_compare.csv", np.column_stack([real_t, real_pen, sim_pen]),
               delimiter=",", header="time_s,real_hanging_rad,sim_hanging_rad", comments="")
    report = dict(params_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
                  recording=(recording or free_files[0]).name, samples=n,
                  angle_rmse_rad=float(np.sqrt(np.mean((sim_pen - real_pen)**2))),
                  physical_parameters_refitted=False)
    (out_dir / "freeswing_compare.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def plot_motor_sweep(npz_path: Path, *, out_dir: Path) -> None:
    """Plot motor target vs actual during the ±90° sanity sweep.

    Recordings with per-segment timestamps are converted to cumulative time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = dict(np.load(npz_path))
    t = d["t"].copy()

    # Cumulative-time backfill for legacy per-segment recordings.
    # If t drops backwards between samples, every drop marks the start
    # of a new move_motor_to call. Use the previous sample's cumulative
    # value as the new offset.
    dt = np.diff(t)
    if (dt < 0).any():
        offset = 0.0
        prev = 0.0
        for i in range(len(t)):
            if i > 0 and d["t"][i] < d["t"][i - 1]:
                offset = prev
            t[i] = d["t"][i] + offset
            prev = t[i]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, np.degrees(d["motor"]),  label="motor (firmware-reported)")
    axes[0].plot(t, np.degrees(d["target"]), "--", label="target", alpha=0.7)
    axes[0].axhline(+90, color="k", lw=0.3); axes[0].axhline(-90, color="k", lw=0.3)
    axes[0].axhline(0, color="k", lw=0.2)
    axes[0].set_ylabel("angle (°)"); axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Motor ±90° sanity sweep — firmware tracks commanded angle")
    axes[1].plot(t, d["accel_cmd"], label="commanded accel (rad/s²)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("accel (rad/s²)")
    axes[1].grid(alpha=0.3); axes[1].legend(loc="upper right")
    plt.tight_layout()
    fig_path = out_dir / "motor_sweep.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(_dim(f"  -> {fig_path.name}"))


# ---------------------------------------------------------------------------
# Standalone motor-validation entry point (uses last collected dir, or
# records its own one-off motor sweep)
# ---------------------------------------------------------------------------

def validate_motor(args: argparse.Namespace) -> None:
    port = pick_port(args.port)
    out_dir = make_output_dir(args.out_dir)
    if input("Before opening USB, switch 12 V OFF and center the arm. Type OFF: ").strip() != "OFF":
        raise RuntimeError("Motor sweep cancelled")
    with LowLevelClient(port, baud=args.baud, timeout=.1) as client:
        record_motor_sweep(client, out_dir)
    plot_motor_sweep(out_dir / "motor_sweep.npz", out_dir=out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_recordings(in_dir: Path, params: Path, out_dir: str) -> None:
    meta = load_metadata(in_dir)
    if meta.get("collection_state", "complete") != "complete":
        raise ValueError("Validation collection is incomplete")
    document = json.loads(params.read_text())
    sources = document.get("run_meta", {}).get("training_sources")
    if not sources:
        raise ValueError("Refit archived data with this release to record training provenance")
    training = {item["sha256"] for item in sources}
    recordings = sorted(in_dir.glob("free_run_*.npz"))
    if not recordings:
        raise ValueError("No validation recordings")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in recordings]
    if training.intersection(hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("Validation recordings must be independent of fitting trials")
    output = make_output_dir(out_dir)
    reports = []
    for path in recordings:
        destination = output / path.stem
        destination.mkdir()
        reports.append(plot_free_swing_compare(in_dir, document["pendulum"]["derived"],
                       out_dir=destination, params_path=params, recording=path))
    (output / "validation.json").write_text(json.dumps(reports, indent=2) + "\n")


def configure_parser(p) -> None:
    sub = p.add_subparsers(dest="cmd")
    from . import sysid_accel, motor_envelope_probe, fit_motor_envelope_coupling
    for name, module, help_text in (
        ("accel", sysid_accel, "acceleration step/chirp capture and offline replay"),
        ("motor-envelope", motor_envelope_probe, "attended feedback-driven motor envelope"),
        ("fit-coupling", fit_motor_envelope_coupling, "fit driven coupling from envelope traces"),
    ):
        module.configure_parser(sub.add_parser(name, help=help_text))

    def common_device(sp):
        sp.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
        sp.add_argument("--baud", type=int, default=500_000)
        sp.add_argument("--out-dir", default=None,
                        help="output directory (auto-timestamped if omitted)")

    pc = sub.add_parser("collect", help="acquire data only")
    common_device(pc)
    pc.add_argument("--n-free-runs", type=int, default=DEFAULT_FREE_SWING_RUNS)
    pc.add_argument("--duration-s", type=float, default=DEFAULT_FREE_SWING_DURATION_S)
    pc.add_argument("--sample-hz", type=float, default=100.0)
    pc.add_argument("--skip-motor", action="store_true",
                    help="skip the motor ±90° sanity sweep")

    pf = sub.add_parser("fit", help="fit parameters from a collected directory")
    pf.add_argument("--in-dir", required=True, type=Path)
    pf.add_argument("--out-json", required=True, type=Path,
                    help="where to write the fitted parameters, e.g. "
                         "sysid_params_rig1.json. Required rather than "
                         "defaulted: friction is per-rig, so writing to a "
                         "generic filename is how one rig's measurements end "
                         "up silently describing another.")

    pv = sub.add_parser("validate-motor", help="motor ±90° sanity sweep only")
    common_device(pv)

    check = sub.add_parser("validate", help="compare independent recordings with a frozen fitted model")
    check.add_argument("--in-dir", type=Path, required=True)
    check.add_argument("--params", type=Path, required=True)
    check.add_argument("--out-dir", required=True)

    # Default (no subcommand): full pipeline.
    common_device(p)
    p.add_argument("--n-free-runs", type=int, default=DEFAULT_FREE_SWING_RUNS)
    p.add_argument("--duration-s", type=float, default=DEFAULT_FREE_SWING_DURATION_S)
    p.add_argument("--sample-hz", type=float, default=100.0)
    p.add_argument("--skip-motor", action="store_true")
    p.add_argument("--out-json", type=Path,
                   help="where to write the fitted parameters, e.g. "
                        "sysid_params_rig1.json. Required rather than "
                        "defaulted: friction is per-rig, so writing to a "
                        "generic filename is how one rig's measurements end "
                        "up silently describing another.")

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    configure_parser(p)
    return p.parse_args(argv)


def execute(args: argparse.Namespace) -> int:
    if args.cmd in ("accel", "motor-envelope", "fit-coupling"):
        from . import sysid_accel, motor_envelope_probe, fit_motor_envelope_coupling
        return {"accel": sysid_accel, "motor-envelope": motor_envelope_probe,
                "fit-coupling": fit_motor_envelope_coupling}[args.cmd].execute(args)
    if args.cmd is None and args.out_json is None:
        raise ValueError("Full pipeline requires --out-json")
    if args.cmd in (None, "collect", "validate-motor"):
        if args.baud != PROFILE.baud:
            raise ValueError("Use the lab firmware baud rate: 500000")
    if args.cmd in (None, "collect"):
        if (not 1 <= args.n_free_runs <= 20 or not math.isfinite(args.duration_s)
                or not 3 <= args.duration_s <= 60 or not math.isfinite(args.sample_hz)
                or not 20 <= args.sample_hz <= 100):
            raise ValueError("Use 1-20 trials, 3-60 seconds, and 20-100 Hz sampling")
        if args.cmd is None and args.out_json.exists():
            raise FileExistsError(args.out_json)
    if args.cmd == "collect":
        collect_data(args)
    elif args.cmd == "fit":
        fit_data(args.in_dir, args.out_json)
    elif args.cmd == "validate-motor":
        validate_motor(args)
    elif args.cmd == "validate":
        validate_recordings(args.in_dir, args.params, args.out_dir)
    else:
        # Full pipeline
        out_dir = collect_data(args)
        fit_data(out_dir, args.out_json)
        print(_bold("\nDone. Use the policy training scripts next:"))
        print(f"  Pass --params {args.out_json} to pendulum-lab sim or train.")
    return 0


def main(argv: list[str] | None = None) -> int:
    return execute(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
