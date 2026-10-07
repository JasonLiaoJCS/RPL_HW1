import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pendulum_lab.cli import main, parser
from pendulum_lab.contracts import PARAMS, PROFILE, checked_action, observation_frame, state_from_obs
from pendulum_lab.controllers import ALGORITHMS, TRAINABLE, load_controller
from pendulum_lab.environment import make_env, true_state
from pendulum_lab.evaluation import summarize
from pendulum_lab.hardware import run, validate_state
from pendulum_lab.plant.pendulum_env import PendulumParams
from pendulum_lab.serial_client import State


class InfrastructureTests(unittest.TestCase):
    def test_starter_parameters_are_synthetic_and_physically_valid(self):
        doc = json.loads(PARAMS.read_text())
        self.assertEqual(doc["run_meta"]["parameter_status"], "synthetic_unidentified_starter")
        self.assertNotIn("fit", doc["pendulum"])
        self.assertNotIn("collected_at", doc["run_meta"])
        p = PendulumParams.load(PARAMS)
        self.assertTrue(np.isfinite(list(vars(p).values())).all())
        self.assertGreater(p.pendulum_mass_kg, 0)
        self.assertGreater(p.pendulum_inertia_kg_m2, p.pendulum_mass_kg * p.pendulum_com_m**2)
        self.assertGreaterEqual(p.pendulum_friction, 0)
        self.assertGreaterEqual(p.pendulum_coulomb, 0)
        env = make_env(PARAMS, duration=0.1)
        try:
            self.assertEqual(env.unwrapped.params, p)
            env.reset(seed=0)
            for _ in range(5):
                obs, reward, _, _, _ = env.step(np.array([0.25]))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
        finally:
            env.close()

    def test_physical_assets_are_complete(self):
        assets = Path(__file__).resolve().parents[1] / "hardware"
        manifest = json.loads((assets / "assets.json").read_text())
        for name, digest in manifest["sha256"].items():
            with self.subTest(asset=name):
                self.assertEqual(hashlib.sha256((assets / name).read_bytes()).hexdigest(), digest)

    def test_angle_action_and_history_contract(self):
        x = [0.2, -0.3, 0.4, -0.5]
        np.testing.assert_allclose(state_from_obs(observation_frame(x, 0.1)), x, atol=1e-7)
        self.assertEqual(checked_action(9), 1)
        for invalid in (float("nan"), float("inf"), [0, 1]):
            with self.assertRaises(ValueError):
                checked_action(invalid)

    def test_simulation_is_seeded_and_centered(self):
        env = make_env(PARAMS, duration=0.1, start="upright")
        try:
            first, _ = env.reset(seed=123)
            second, _ = env.reset(seed=123)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (24,))
            self.assertLessEqual(abs(true_state(env)[0]), math.radians(5))
            for _ in range(5):
                obs, reward, terminated, truncated, _ = env.step(np.array([0.0]))
                self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_physical_arm_guard_terminates_simulation(self):
        env = make_env(PARAMS)
        try:
            self.assertEqual(PROFILE.arm_limit_deg, 100)
            for degrees in (-101, -99, -90, 90, 99, 101):
                with self.subTest(degrees=degrees):
                    env.reset(seed=0)
                    angle = math.radians(degrees)
                    env.unwrapped.data.qpos[env.unwrapped._motor_qpos_addr] = angle
                    env.unwrapped.set_actuator_state(angle)
                    env.unwrapped.seed_measurement_ring()
                    _, _, terminated, _, info = env.step(np.array([0.0]))
                    outside = abs(degrees) > PROFILE.arm_limit_deg
                    self.assertEqual(terminated, outside)
                    self.assertEqual(bool(info.get("lab_guard")), outside)
        finally:
            env.close()

    def test_only_complete_final_ten_seconds_pass(self):
        rows = [[i / 50, 0, 0, 0, 0, 0] for i in range(750)]
        self.assertTrue(summarize(rows, "duration", 15)["success"])
        self.assertFalse(summarize(rows[:150], "duration", 3)["success"])
        self.assertFalse(summarize(rows, "arm_guard", 15)["success"])
        rows[-1][4] = 8
        self.assertFalse(summarize(rows, "duration", 15)["success"])

    def test_cli_saves_trace_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out = Path(d) / "run"
            args = ["sim", "--algorithm", "zero", "--duration", "0.1", "--output", str(out)]
            self.assertEqual(main(args), 0)
            self.assertEqual(np.load(out / "episode_000.npz")["samples"].shape, (5, 6))
            self.assertFalse(json.loads((out / "summary.json").read_text())[0]["success"])
            self.assertEqual(main(args), 1)

    def test_cli_saves_simulation_gif(self):
        import mujoco
        from PIL import Image
        try:
            mujoco.Renderer(make_env(PARAMS, duration=0.1).unwrapped.model, 24, 32).close()
        except Exception as exc:
            self.skipTest(f"no offscreen OpenGL: {exc}")
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / "run"
            self.assertEqual(main(["sim", "--algorithm", "zero", "--duration", "0.1",
                                   "--gif", "--output", str(out)]), 0)
            with Image.open(out / "episode_000.gif") as gif:
                # Pillow merges identical frames but keeps their total duration.
                durations = []
                for i in range(gif.n_frames):
                    gif.seek(i)
                    durations.append(gif.info["duration"])
                self.assertEqual(sum(durations), 3 * 2 * round(1000 * PROFILE.dt))

    def test_hardware_duration_allows_demo_length_runs(self):
        with self.assertRaisesRegex(ValueError, r"\(0, 30\]"):
            run(load_controller("zero", PARAMS), "fake", 30.1, [])

    def test_all_student_controllers_are_explicit_exercises(self):
        for name in ALGORITHMS[1:]:
            c = load_controller(name, PARAMS)
            self.assertEqual(c.__class__.__module__, f"pendulum_lab.algorithms.{name}")
            c.reset()
            with self.assertRaises(NotImplementedError):
                c.act(np.zeros(24))

    def test_training_cli_reaches_algorithm_stubs(self):
        for name in TRAINABLE:
            with self.subTest(algorithm=name), tempfile.TemporaryDirectory() as d:
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    result = main(["train", "--algorithm", name, "--output", str(Path(d) / "run")])
                self.assertEqual(result, 1)
                self.assertIn(f"NotImplementedError: Implement {name.upper()} training", errors.getvalue())

    def test_custom_algorithm_is_available_in_each_runner(self):
        for command in ("sim", "hardware", "train"):
            args = [command, "--algorithm", "custom", "--output", "unused"]
            if command == "hardware":
                args.extend(["--port", "fake"])
            self.assertEqual(parser().parse_args(args).algorithm, "custom")
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / "run"
            self.assertEqual(main(["sim", "--algorithm", "custom", "--output", str(out)]), 1)
            report = json.loads((out / "summary.json").read_text())[0]
            self.assertIn("algorithms/custom.py", report["stop_reason"])
            self.assertEqual(report["samples"], 0)


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Client:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.engaged = False
        self.closed = False
        self.timestamp = 0
        self.commands = []
        self.ever_engaged = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disengage_motor()
        self.closed = True

    def wait_until_ready(self):
        return True

    def require_profile(self, **expected):
        if expected != dict(microsteps=32, baud=500000, protocol_version=3):
            raise AssertionError("Release profile not required")
        if self.failure == "profile":
            raise RuntimeError("profile mismatch")

    def get_state(self):
        self.timestamp += 20000
        if self.engaged and self.failure == "serial":
            raise OSError("disconnected")
        arm = math.radians(PROFILE.arm_limit_deg + 1) if self.engaged and self.failure == "arm" else 0
        return State(self.timestamp, arm, 0.0, 0.0, 0.0)

    def engage_motor(self):
        self.engaged = self.ever_engaged = True

    def disengage_motor(self):
        self.engaged = False

    def get_motor_status(self):
        return SimpleNamespace(motor_engaged=self.engaged and self.failure != "watchdog")

    def get_safety_faults(self):
        return 0

    def set_acceleration(self, action):
        self.commands.append(action)


class HardwareTests(unittest.TestCase):
    def test_arm_travel_boundary_and_centered_start(self):
        for degrees in (-99, -90, 90, 99):
            state = State(100, math.radians(degrees), 0, 0, 0)
            validate_state(state)
            with self.assertRaisesRegex(RuntimeError, "centered"):
                validate_state(state, startup=True)
        for degrees in (-101, -100, 100, 101):
            with self.subTest(degrees=degrees), self.assertRaisesRegex(RuntimeError, "Arm travel"):
                validate_state(State(100, math.radians(degrees), 0, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "Arm travel/speed"):
            validate_state(State(100, 0, 0, PROFILE.arm_speed_limit + .1, 0))

    def exercise(self, client, controller=None, confirm="RUN"):
        clock = Clock()
        with patch("pendulum_lab.hardware.time", clock):
            rows = []
            run(controller or load_controller("zero", PARAMS), "fake", 0.1, rows,
                client_factory=lambda *a, **kw: client, confirm=lambda _: confirm)
            return rows

    def test_happy_path_disengages_and_logs(self):
        client = Client()
        rows = self.exercise(client)
        self.assertEqual(len(rows), 5)
        self.assertEqual(client.commands, [0] * 5)
        self.assertTrue(client.closed)
        self.assertFalse(client.engaged)

    def test_failures_disengage_and_never_reengage(self):
        for reason in ("profile", "serial", "arm", "watchdog"):
            client = Client(failure=reason)
            with self.assertRaises((RuntimeError, OSError)):
                self.exercise(client)
            self.assertTrue(client.closed)
            self.assertFalse(client.engaged)
            self.assertEqual(client.commands, [])
            if reason == "profile":
                self.assertFalse(client.ever_engaged)

    def test_unimplemented_controller_cannot_open_usb(self):
        client = Client()
        with self.assertRaises(NotImplementedError):
            self.exercise(client, load_controller("pid", PARAMS))
        self.assertFalse(client.ever_engaged)
        self.assertFalse(client.closed)  # never entered the client context

    def test_late_controller_cannot_send_command(self):
        clock = Clock()
        class Slow:
            calls = 0
            def reset(self):
                pass
            def act(self, obs):
                self.calls += 1
                if self.calls > 1:
                    clock.sleep(0.1)
                return 0.5
        client = Client()
        with patch("pendulum_lab.hardware.time", clock), self.assertRaisesRegex(RuntimeError, "deadline"):
            run(Slow(), "fake", 0.1, [], client_factory=lambda *a, **kw: client,
                confirm=lambda _: "RUN")
        self.assertEqual(client.commands, [])
        self.assertFalse(client.engaged)

    def test_stale_and_invalid_feedback(self):
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            validate_state(State(100, 0, 0, 0, 0), 100)
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            validate_state(State(100, 0, float("nan"), 0, 0))
        # A normal uint32 micros rollover is not a stale frame.
        validate_state(State(10000, 0, 0, 0, 0), 2**32 - 10000)


if __name__ == "__main__":
    unittest.main()
