"""Acceleration step/chirp capture and replay with explicit lab limits."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from .contracts import PROFILE
from .plant.pendulum_env import RotaryInvertedPendulumEnv
from .sysid_driven import DrivenSampler, check_settings, powered_session


def waveform_step(t: float) -> float:
    """Unscaled bipolar acceleration pulses, in rad/s².

    The runner applies an explicit scale to meet the lab acceleration cap.
    Sampling and actuator delay can leave residual velocity and displacement;
    paired pulses do not guarantee a physical return to rest.
    """
    # Settle
    if t < 0.3: return 0.0
    # Pulse 1: unscaled amplitude 50, T=0.1s
    if t < 0.4: return +50.0
    if t < 0.5: return -50.0
    if t < 1.0: return 0.0
    # Pulse 2: amplitude 100, T=0.05s
    if t < 1.05: return +100.0
    if t < 1.10: return -100.0
    if t < 1.6:  return 0.0
    # Pulse 3: amplitude 150, T=0.033s
    if t < 1.633: return +150.0
    if t < 1.667: return -150.0
    if t < 2.2:   return 0.0
    # Pulse 4: negative-side first
    if t < 2.25:  return -100.0
    if t < 2.30:  return +100.0
    if t < 2.8:   return 0.0
    # Reversal segment for comparing commanded and observed velocity changes.
    if t < 2.85:  return +100.0
    if t < 2.95:  return -100.0
    if t < 3.00:  return +100.0
    return 0.0


STEP_DURATION_S = 3.5


def waveform_chirp(t: float) -> float:
    """Unscaled chirp with instantaneous frequency from 0.5 to 5.5 Hz.

    The coefficient in the phase increases to 3 Hz, but differentiating the
    phase gives 5.5 Hz at the endpoint.
    """
    if t < 0.3: return 0.0
    s = t - 0.3
    if s > 8.0: return 0.0
    f0, f1 = 0.5, 3.0
    f = f0 + (f1 - f0) * s / 8.0
    return 100.0 * math.sin(2.0 * math.pi * f * s)


CHIRP_DURATION_S = 9.0



def configure_parser(parser):
    parser.add_argument("--port")
    parser.add_argument("--waveform", choices=("step", "chirp"), default="step")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--sample-hz", type=float, default=200.0)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-real", action="store_true")
    mode.add_argument("--replay", type=Path, help="replay a completed capture offline")


def replay(command_t, commands, times, initial, params):
    """Replay actual command times, quantized to the 5 ms simulator tick."""
    command_t = np.asarray(command_t, dtype=float)
    commands = np.asarray(commands, dtype=float)
    times = np.asarray(times, dtype=float)
    if (len(command_t) != len(commands) or not len(command_t) or not len(times)
            or not np.isfinite(np.r_[command_t, commands, times, initial]).all()
            or np.any(np.diff(command_t) <= 0) or np.any(np.diff(times) <= 0)
            or command_t[0] < 0 or times[0] < 0
            or np.max(np.abs(commands)) > PROFILE.max_accel):
        raise ValueError("Invalid command/state timeline or acceleration")
    if times[-1] > 60:
        raise ValueError("Replay duration exceeds 60 seconds")
    dt = .005
    env = RotaryInvertedPendulumEnv(
        params_path=params, control_freq_hz=1/dt, action_mode="accel",
        action_smooth_window=1, max_accel_rad_s2=PROFILE.max_accel,
        motor_max_accel_rad_s2=PROFILE.max_accel, transport_delay_s=.010,
        motor_planner_speed_limit_rad_s=5.0, firmware_obs_model=True,
        episode_length_s=float(times[-1])+1, terminate_on_spin_loop=False,
    )
    try:
        env.reset(seed=0)
        m, p, mv, pv = initial
        env.data.qpos[env._motor_qpos_addr] = m
        env.data.qpos[env._pen_qpos_addr] = p
        env.data.qvel[env._motor_qvel_addr] = mv
        env.data.qvel[env._pen_qvel_addr] = pv
        env.set_actuator_state(m, planner_velocity_rad_s=mv)
        mujoco.mj_forward(env.model, env.data)
        env.seed_measurement_ring()
        history = [(0., m, p)]
        for i in range(math.ceil(times[-1]/dt)):
            t = i*dt
            index = np.searchsorted(command_t, t + 1e-12, side="right")-1
            accel = commands[index] if index >= 0 else 0.
            _, _, terminated, _, info = env.step(np.array([accel/PROFILE.max_accel], dtype=np.float32))
            history.append(((i+1)*dt, info["motor_pos"], info["phi"]))
            if terminated:
                raise RuntimeError("Simulator terminated before replay finished")
        history = np.asarray(history)
        return np.column_stack([np.interp(times, history[:, 0], history[:, j]) for j in (1, 2)])
    finally:
        env.close()


def plot_comparison(t, accel, real, sim, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].step(t, accel, where="post", label="applied command")
    for j, label in enumerate(("arm angle (rad)", "hanging-zero angle (rad)")):
        if real is not None:
            axes[j+1].plot(t, real[:, j], label="recorded")
        axes[j+1].plot(t, sim[:, j], label="simulated")
        axes[j+1].set_ylabel(label)
    axes[0].set_ylabel("acceleration (rad/s^2)")
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def execute(args):
    check_settings(args.sample_hz)
    fn, duration, default_scale = (
        (waveform_step, STEP_DURATION_S, 1/3) if args.waveform == "step"
        else (waveform_chirp, CHIRP_DURATION_S, .1))
    scale = default_scale if args.scale is None else args.scale
    peak = 150 if args.waveform == "step" else 100
    if not math.isfinite(scale) or not 0 < scale <= PROFILE.max_accel/peak:
        raise ValueError("Waveform scale must be positive and respect 50 rad/s^2")
    # Load the requested plant before opening USB.
    from .plant.pendulum_env import PendulumParams
    PendulumParams.load(args.params)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {"waveform": args.waveform, "scale": scale,
              "command_hz": PROFILE.hz, "sample_hz": args.sample_hz,
              "params": str(args.params), "sim_only": args.skip_real,
              "complete": False, "replay_tick_s": .005,
              "angle_frame": "firmware hanging-zero, no sign inversion"}
    (args.out_dir/"metadata.json").write_text(json.dumps(report, indent=2)+"\n")
    initial = np.zeros(4)
    real = None
    if args.replay:
        with np.load(args.replay, allow_pickle=False) as rec:
            if not bool(rec["complete"]):
                raise ValueError("Cannot replay an incomplete capture as a completed experiment")
            rows = rec["samples"]
            times = rows[:, 0] - float(rec["start_host_time_s"])
            command_t, commands = rec["command_t"], rec["accel_commands"]
            initial = rec["initial_state"]
            real = rows[:, 3:5]
            report.update(waveform=str(rec["waveform"]), scale=float(rec["scale"]),
                          source=str(args.replay))
    elif args.skip_real:
        command_t = np.arange(0, duration, PROFILE.dt)
        commands = np.array([scale*fn(t) for t in command_t])
        times = np.arange(0, duration, 1/args.sample_hz)
    else:
        if not args.port:
            raise ValueError("--port is required for capture")
        with powered_session(args.port) as client:
            s = client.get_state()
            initial = np.array([s.motor_pos_rad, s.pendulum_pos_rad,
                                s.motor_vel_rad_s, s.pendulum_vel_rad_s])
            sampler = DrivenSampler(client, args.sample_hz)
            complete = False
            try:
                for _ in range(round(duration*args.sample_hz)):
                    sampler.sample(lambda _: scale*fn(
                        time.monotonic()-sampler.start), 0)
                complete = True
            finally:
                client.disengage_motor()
                sampler.save(args.out_dir/"capture.npz", complete=complete,
                             initial_state=initial, waveform=args.waveform, scale=scale)
        rows = np.asarray(sampler.rows)
        times = rows[:, 0]-sampler.start
        command_t, commands = np.asarray(sampler.command_times), np.asarray(sampler.commands)
        real = rows[:, 3:5]
    sim = replay(command_t, commands, times, initial, args.params)
    indices = np.searchsorted(command_t, times, side="right")-1
    accel = np.where(indices >= 0, commands[np.maximum(indices, 0)], 0.)
    # Empty real arrays identify a simulation-only comparison archive.
    np.savez_compressed(args.out_dir/"comparison.npz", t=times, accel_cmd=accel,
                        real_motor=real[:, 0] if real is not None else [],
                        real_pen=real[:, 1] if real is not None else [],
                        sim_motor=sim[:, 0], sim_pen=sim[:, 1], waveform=report["waveform"])
    plot_comparison(times, accel, real, sim, args.out_dir/"comparison.png")
    if real is not None:
        errors = sim-real
        errors[:, 1] = (errors[:, 1]+np.pi) % (2*np.pi)-np.pi
        report["motor_rmse_rad"], report["pendulum_wrapped_rmse_rad"] = (
            np.sqrt(np.mean(errors**2, axis=0)).tolist())
    report["complete"] = True
    (args.out_dir/"metadata.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0
