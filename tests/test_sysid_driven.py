import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pendulum_lab.cli import parser
from pendulum_lab.contracts import PARAMS
from pendulum_lab import sysid_accel as accel, sysid_driven as driven
from pendulum_lab import motor_envelope_probe as envelope
from pendulum_lab import fit_motor_envelope_coupling as coupling


def trace(path, complete=True):
    t = np.arange(400)/200
    rows = np.column_stack((t, np.repeat([0, 1], 200), np.repeat([25., -25.], 200),
                            .15*np.sin(3*t), .3*np.sin(9*t)+.05*np.cos(2*t),
                            .45*np.cos(3*t), 2.7*np.cos(9*t)-.1*np.sin(2*t)))
    np.savez(path, samples=rows, complete=complete)
    return rows


class NumericalTests(unittest.TestCase):
    def test_waveform_values(self):
        self.assertEqual([accel.waveform_step(t) for t in
                          [0, .35, .45, 1.025, 1.075, 1.62, 1.65, 2.225, 2.275, 2.825, 2.9, 2.975, 3.1]],
                         [0, 50, -50, 100, -100, 150, -150, -100, 100, 100, -100, 100, 0])
        for t in [.3, 1., 2., 4., 8.3]:
            s = t-.3
            self.assertAlmostEqual(accel.waveform_chirp(t), 100*math.sin(2*math.pi*(.5+2.5*s/8)*s))

    def test_known_coefficients(self):
        rng = np.random.default_rng(12)
        x = np.column_stack([rng.normal(size=(200, 2)), np.ones(200)])
        result = coupling._fit([(x, x@np.array([-2., .3, .1]))])
        np.testing.assert_allclose([result[k] for k in
                                   ["tangential_coupling", "centrifugal_coupling", "constant_bias_rad_s2"]],
                                  [-2., .3, .1], atol=1e-12)
        self.assertAlmostEqual(result["r2"], 1.)
        with self.assertRaises(ValueError):
            coupling._fit([])
        with self.assertRaises(ValueError):
            coupling._fit([(np.ones((10, 3)), np.arange(10))])

    def test_fit_cli_and_invalid_captures(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            rows = trace(root/"trace.npz")
            params = json.loads(PARAMS.read_text())
            params["pendulum"]["fit"] = {"omega_n_small_amp_rad_s": math.sqrt(150)}
            (root/"rig.json").write_text(json.dumps(params))
            args = parser().parse_args(["sysid", "fit-coupling", str(root), "--params", str(root/"rig.json")])
            coupling.execute(args)
            report = json.loads((root/"coupling_fit.json").read_text())
            self.assertEqual(report["selected_window_max_samples"], 7)
            self.assertEqual(len(report["window_sensitivity"]), 5)
            with self.assertRaises(FileExistsError):
                coupling.execute(args)
            trace(root/"trace.npz", complete=False)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                coupling._parts_for_trace(root/"trace.npz", window_max=7, gravity_w2=150, damping=.2)
            rows[10, 0] = rows[9, 0]
            np.savez(root/"trace.npz", samples=rows)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                coupling._parts_for_trace(root/"trace.npz", window_max=7, gravity_w2=150, damping=.2)

    def test_sim_only_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(["sysid", "accel", "--skip-real", "--params", str(PARAMS),
                                       "--out-dir", str(root/"sim")])
            with patch.object(accel, "powered_session", side_effect=AssertionError("USB opened")):
                self.assertEqual(accel.execute(args), 0)
            with np.load(root/"sim/comparison.npz") as data:
                self.assertEqual(len(data["real_motor"]), 0)
                self.assertGreater(np.ptp(data["sim_motor"]), .01)
                self.assertLessEqual(np.max(np.abs(data["accel_cmd"])), 50.)
            t = np.arange(1, 101)/200
            cmd_t = np.arange(0, .5, .02)
            commands = np.where(cmd_t < .2, 5., -5.)
            sim = accel.replay(cmd_t, commands, t, [0, 0, 0, 0], PARAMS)
            rows = np.column_stack([t+10, np.zeros((len(t), 2)), sim, np.zeros((len(t), 2))])
            np.savez(root/"capture.npz", samples=rows, command_t=cmd_t,
                     accel_commands=commands, start_host_time_s=10, initial_state=np.zeros(4),
                     waveform="step", scale=.1, complete=True)
            args = parser().parse_args(["sysid", "accel", "--replay", str(root/"capture.npz"),
                                       "--params", str(PARAMS), "--out-dir", str(root/"replay")])
            with patch.object(accel, "powered_session", side_effect=AssertionError("USB opened")):
                accel.execute(args)
            report = json.loads((root/"replay/metadata.json").read_text())
            self.assertLess(report["motor_rmse_rad"], 1e-10)
            self.assertLess(report["pendulum_wrapped_rmse_rad"], 1e-10)


class Clock:
    now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, delay):
        self.now += delay


class FakeClient:
    def __init__(self):
        self.t = 0
        self.angle = 0.
        self.fault = 0
        self.engaged = True
        self.commands = []
        self.disables = 0
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.disengage_motor()
    def get_state(self):
        self.t += 5000
        return SimpleNamespace(time_us=self.t, motor_pos_rad=self.angle,
                               pendulum_pos_rad=0., motor_vel_rad_s=0., pendulum_vel_rad_s=0.)
    def get_safety_faults(self):
        return self.fault
    def get_motor_status(self):
        return SimpleNamespace(motor_engaged=self.engaged)
    def set_acceleration(self, value):
        self.commands.append(value)
    def engage_motor(self):
        self.engaged = True
    def disengage_motor(self):
        self.disables += 1
        self.engaged = False


class GuardTests(unittest.TestCase):
    def test_rejected_settings_before_usb(self):
        with tempfile.TemporaryDirectory() as directory:
            for scale in [1., float("nan"), -1.]:
                args = parser().parse_args(["sysid", "accel", "--scale", str(scale),
                                           "--params", str(PARAMS), "--out-dir", directory])
                with self.assertRaises(ValueError):
                    accel.execute(args)
            args = parser().parse_args(["sysid", "motor-envelope", "--port", "fake",
                                       "--accels", "100", "--out-dir", directory])
            with self.assertRaises(ValueError):
                envelope.execute(args)

    def test_sampler_commands_and_phase_timing(self):
        clock, client = Clock(), FakeClient()
        with patch.object(driven, "time", clock):
            sampler = driven.DrivenSampler(client, 200)
            for _ in range(20):
                sampler.sample(10., 2)
        self.assertTrue(5 <= len(client.commands) <= 6)
        self.assertEqual(sampler.rows[0][1:3], (-1, 0.))
        self.assertEqual(sampler.rows[1][1:3], (2, 10.))
        self.assertTrue(np.all(np.diff(sampler.command_times) >= .015-1e-12))

    def test_fault_watchdog_travel_timing_stale_and_cap(self):
        for failure in ["fault", "watchdog", "travel", "late", "stale", "cap"]:
            with self.subTest(failure=failure):
                clock, client = Clock(), FakeClient()
                with patch.object(driven, "time", clock):
                    sampler = driven.DrivenSampler(client, 200)
                    if failure == "fault": client.fault = 2
                    if failure == "watchdog": client.engaged = False
                    if failure == "travel": client.angle = math.radians(65)
                    if failure == "late": clock.now = 1.
                    if failure == "stale": sampler.previous = 5000
                    with self.assertRaises((RuntimeError, ValueError)):
                        sampler.sample(100 if failure == "cap" else 10., 0)
                    self.assertEqual(client.commands, [])

    def test_session_cleanup_and_no_implicit_consent(self):
        client = FakeClient()
        answers = iter(["OFF", "RUN"])
        with patch.object(driven, "preflight"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "test fault"):
                with driven.powered_session("fake", confirm=lambda _: next(answers),
                                            client_factory=lambda *a, **k: client):
                    raise RuntimeError("test fault")
            self.assertFalse(client.engaged)
        with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
            with driven.powered_session("fake", confirm=lambda _: "", client_factory=lambda *a, **k: self.fail("USB opened")):
                pass

    def test_failed_capture_is_saved(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            client = FakeClient()
            @contextlib.contextmanager
            def session(*args):
                yield client
            args = parser().parse_args(["sysid", "accel", "--port", "fake", "--params", str(PARAMS),
                                       "--out-dir", str(Path(directory)/"out")])
            with patch.object(accel, "powered_session", session), patch.object(driven.DrivenSampler, "sample", side_effect=RuntimeError("fault")):
                with self.assertRaises(RuntimeError):
                    accel.execute(args)
            with np.load(args.out_dir/"capture.npz") as rec:
                self.assertFalse(bool(rec["complete"]))
            self.assertFalse(client.engaged)

    def test_envelope_phase_sequence_and_bounded_return(self):
        calls = []
        sampler = SimpleNamespace()
        with patch.object(envelope, "_sample_until", side_effect=lambda *a, **kw: calls.append(kw)), patch.object(envelope, "_return_to_zero") as return_zero:
            envelope.run_level(sampler, speed=1.5, accel=25., cycles=2)
        self.assertEqual([c["accel"] for c in calls], [25., -25., -25., 25.])
        self.assertEqual([c["phase"] for c in calls], [0, 1, 2, 3])
        for c, velocity in zip(calls, [1.5, 0., -1.5, 0.]):
            self.assertTrue(c["done"](SimpleNamespace(motor_vel_rad_s=velocity)))
        self.assertEqual(return_zero.call_count, 2)
        clock, client = Clock(), FakeClient()
        with patch.object(driven, "time", clock), patch.object(envelope, "time", clock):
            sampler = driven.DrivenSampler(client, 200)
            envelope._return_to_zero(sampler)
        self.assertGreaterEqual(clock.now, .3)
        self.assertTrue(all(abs(a) <= 40 for a in client.commands))
        self.assertTrue(all(row[1] == -1 for row in sampler.rows))

    def test_envelope_rejection_stops_ladder_and_saves_partial(self):
        for fail in [False, True]:
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                client = FakeClient()
                @contextlib.contextmanager
                def session(*args):
                    yield client
                def run(sampler, **kwargs):
                    sampler.rows.append((0., 0., 25., 0., 0., 1.5, 0.))
                    if fail:
                        raise RuntimeError("motion fault")
                args = parser().parse_args(["sysid", "motor-envelope", "--port", "fake",
                                           "--out-dir", str(Path(directory)/"out")])
                with patch.object(envelope, "powered_session", session), patch.object(envelope, "run_level", side_effect=run) as levels, patch("builtins.input", return_value="NO"):
                    with self.assertRaises(RuntimeError):
                        envelope.execute(args)
                self.assertEqual(levels.call_count, 1)
                self.assertFalse(client.engaged)
                summary = json.loads((args.out_dir/"summary.json").read_text())
                self.assertFalse(summary["complete"])
                self.assertIsNone(summary["recommendation_source"])
                with np.load(next(args.out_dir.glob("*.npz"))) as rec:
                    self.assertEqual(bool(rec["complete"]), not fail)
