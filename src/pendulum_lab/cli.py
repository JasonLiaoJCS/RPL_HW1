"""Run from any directory after installing the package with pip install -e ."""
import argparse
import json
from pathlib import Path
import signal
import sys
import numpy as np

from .contracts import PARAMS, PROFILE, checked_action, manifest
from .controllers import ALGORITHMS, TRAINABLE, load_controller, algorithm_module
from .environment import make_env, true_state
from .evaluation import COLUMNS, summarize


def parser():
    p = argparse.ArgumentParser(description="Rotary pendulum homework tools")
    sub = p.add_subparsers(dest="command", required=True)
    from .sysid import add_parser
    add_parser(sub)
    sub.add_parser("ports", help="list USB serial devices without opening them")
    for name in ("preflight", "motor-check"):
        s = sub.add_parser(name)
        s.add_argument("--port", required=True)
    for name in ("sim", "hardware", "train"):
        s = sub.add_parser(name)
        s.add_argument("--algorithm", choices=TRAINABLE if name == "train" else ALGORITHMS, required=True)
        s.add_argument("--params", type=Path, default=PARAMS)
        s.add_argument("--config", type=Path)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--duration", type=float, default=3 if name == "hardware" else 15)
        if name != "train":
            s.add_argument("--checkpoint", type=Path)
        if name == "hardware":
            s.add_argument("--port", required=True)
        else:
            s.add_argument("--start", choices=("hanging", "upright", "mixed"), default="hanging")
            s.add_argument("--seed", type=int, default=0)
        if name == "sim":
            s.add_argument("--episodes", type=int, default=1)
            s.add_argument("--render", action="store_true")
            s.add_argument("--gif", action="store_true")
    return p


def simulate(args, controller, rows, episode):
    env = make_env(args.params, duration=args.duration, start=args.start, render=args.render)
    gif = None
    try:
        obs, _ = env.reset(seed=args.seed + episode)
        if args.gif:
            from .media import GifRecorder
            gif = GifRecorder(env)
            gif.capture()
        controller.reset()
        for tick in range(round(args.duration * PROFILE.hz)):
            action = checked_action(controller.act(obs))
            obs, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            rows.append([(tick + 1) * PROFILE.dt, *true_state(env), action])
            if gif:
                gif.capture()
            if args.render:
                env.render()
            if terminated:
                return "arm_guard" if info.get("lab_guard") else "terminated"
            if truncated:
                break
        return "duration"
    finally:
        if gif:
            gif.save(args.output / f"episode_{episode:03d}.gif")
        env.close()


def execute(args):
    if args.command == "sysid":
        from .sysid import execute as execute_sysid
        return execute_sysid(args)
    if args.command == "ports":
        from serial.tools.list_ports import comports
        for p in comports():
            print(f"{p.device}: {p.description} ({p.hwid})")
        return 0
    if args.command in ("preflight", "motor-check"):
        from .hardware import inspect, motor_check
        (inspect if args.command == "preflight" else motor_check)(args.port)
        return 0
    if not np.isfinite(args.duration) or not 0 < args.duration <= (30 if args.command == "hardware" else 60):
        raise ValueError("Invalid duration: hardware maximum 30 s, simulation maximum 60 s")
    if getattr(args, "episodes", 1) < 1:
        raise ValueError("Episodes must be positive")
    metadata = {"algorithm": args.algorithm, "contract": manifest(args.params),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.command == "train":
        config = json.loads(args.config.read_text()) if args.config else {}
        config.setdefault("seed", args.seed)
        def env_factory():
            return make_env(args.params, duration=args.duration, start=args.start)
        algorithm_module(args.algorithm).train(env_factory=env_factory, config=config, output_dir=args.output)
        return 0
    controller = load_controller(args.algorithm, args.params, args.config, args.checkpoint)
    reports = []
    failed = False
    for episode in range(getattr(args, "episodes", 1)):
        rows = []
        reason = "duration"
        try:
            if args.command == "hardware":
                from .hardware import run
                run(controller, args.port, args.duration, rows)
            else:
                reason = simulate(args, controller, rows, episode)
        except (Exception, KeyboardInterrupt) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failed = True
        finally:
            np.savez_compressed(args.output / f"episode_{episode:03d}.npz",
                                samples=np.asarray(rows).reshape(-1, 6), columns=COLUMNS)
            result = summarize(rows, reason, args.duration)
            reports.append(result)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(result))
        if failed:
            break
    if args.command == "hardware":
        print("Run ended. Switch 12 V OFF before touching or recentering the mechanism.")
    return 1 if failed else 0


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(parser().parse_args(argv))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
