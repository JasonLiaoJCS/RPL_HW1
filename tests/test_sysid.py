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

from pendulum_lab.cli import main, parser
from pendulum_lab import sysid_core as core, sysid_wizard as wizard
from pendulum_lab.plant import pendulum_geometry as geometry
from pendulum_lab.plant.pendulum_env import PendulumParams, RotaryInvertedPendulumEnv
from pendulum_lab.serial_client import State


def curves():
    t = np.arange(1001) / 100
    return t, [a * np.exp(-d*t) * np.cos(12.5*t) + .02
               for a, d in [(.65, .13), (.5, .18), (.8, .1)]]


def recordings(root, offset=.2):
    root.mkdir()
    wizard.save_metadata(root, hanging_zero_rad=offset, collection_state="complete")
    t, signals = curves()
    for i, pen in enumerate(signals):
        np.savez(root / f"free_run_{i+1}.npz", t=t, pen=pen+offset,
                 motor=np.zeros_like(t), pvel=np.gradient(pen, t), mvel=np.zeros_like(t))


class IdentificationMathTests(unittest.TestCase):
    def test_synthetic_numerical_fixture(self):
        # Frozen regression values for a synthetic free-swing trace.
        expected = dict(n_extrema_used=39, period_s=.5026315789473684,
                        small_amp_period_s=.4980281958886676, omega_n_rad_s=12.501297892870983,
                        omega_n_small_amp_rad_s=12.61612366337622, zeta=.010730885278751644,
                        rest_angle_rad=.020984869601588355, decay_constant_s_inv=.1341499935238982,
                        r_per_half_cycle=.9668479724743495, delta_c_per_half_cycle_rad=0.,
                        envelope_rmse_combined_rad=.00544752311210967)
        t, signals = curves()
        result = core.fit_free_swing(t, signals[0])
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value, places=9, msg=key)
        derived = core.derive_pendulum_friction(result)
        inertia = geometry.PENDULUM_MASS_KG*geometry.PENDULUM_COM_M**2 + geometry.PENDULUM_I_COM_SWING_KG_M2
        self.assertAlmostEqual(derived["viscous_friction_N_m_s"], 2*result["decay_constant_s_inv"]*inertia)
        self.assertEqual(derived["inertia_predicted_kg_m2"], inertia)
        fits = [core.fit_free_swing(t, signal) for signal in signals]
        aggregate = core.aggregate_free_swing_fits(fits)
        self.assertEqual(aggregate["n_runs"], 3)
        self.assertAlmostEqual(aggregate["small_amp_period_s_stdev"], .002678008024577641)

class OfflineTests(unittest.TestCase):
    def test_npz_fit_export_and_explicit_simulator_parameters(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recordings(root / "train")
            output = root / "rig.json"
            with patch("pendulum_lab.plant.pendulum_env.RotaryInvertedPendulumEnv", wraps=RotaryInvertedPendulumEnv) as factory:
                result = wizard.fit_data(root / "train", output)
            self.assertEqual(factory.call_args.kwargs["params_path"], output)
            doc = json.loads(output.read_text())
            self.assertEqual(doc["pendulum"]["fit"]["n_runs"], 3)
            t, signals = curves()
            expected = core.derive_pendulum_friction(core.aggregate_free_swing_fits([
                core.fit_free_swing(t, signal+.2-.2) for signal in signals]))
            for key in result:
                self.assertAlmostEqual(result[key], expected[key], places=14)
            self.assertEqual(PendulumParams.load(output).pendulum_inertia_kg_m2, result["inertia_measured_kg_m2"])
            self.assertTrue((root/"train/freeswing_compare.png").is_file())
            self.assertTrue((root/"train/freeswing_compare.csv").is_file())
            with self.assertRaises(FileExistsError):
                wizard.fit_data(root/"train", output)
            with self.assertRaisesRegex(ValueError, "independent"):
                wizard.validate_recordings(root/"train", output, str(root/"invalid"))
            recordings(root/"holdout", offset=.3)
            wizard.validate_recordings(root/"holdout", output, str(root/"validation"))
            self.assertEqual(len(json.loads((root/"validation/validation.json").read_text())), 3)

    def test_failed_session_and_moving_arm_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recordings(root/"train")
            wizard.save_metadata(root/"train", collection_state="failed")
            with self.assertRaisesRegex(ValueError, "incomplete or failed"):
                wizard.fit_data(root/"train", root/"rig.json")
            path = root/"train/free_run_1.npz"
            rec = wizard.load_free_recording(path)
            rec["motor"][100] = .2
            np.savez(path, **rec)
            with self.assertRaisesRegex(ValueError, "fixed arm"):
                wizard.load_free_recording(path)

    def test_subcommands_do_not_require_global_out_json(self):
        for command, rest in [("collect", []), ("validate-motor", []),
                              ("fit", ["--in-dir", "data", "--out-json", "rig.json"])]:
            args = parser().parse_args(["sysid", command, *rest])
            self.assertEqual(args.cmd, command)
        with patch.object(wizard, "collect_data") as collect, self.assertRaises(ValueError):
            wizard.execute(parser().parse_args(["sysid"]))
        collect.assert_not_called()
        with patch.object(wizard, "fit_data") as fit, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["sysid", "fit", "--in-dir", "data", "--out-json", "rig.json"]), 0)
        fit.assert_called_once_with(Path("data"), Path("rig.json"))


class Clock:
    now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


class Client:
    def __init__(self, clock, fault=0, frozen=False, stuck=False):
        self.clock, self.fault, self.frozen, self.stuck = clock, fault, frozen, stuck
        self.enabled = False
        self.arm = self.velocity = self.accel = 0.
        self.last_update = self.last_command = clock.now
        self.commands = []
        self.engagements = 0
    def disengage_motor(self):
        self.enabled = False
    def engage_motor(self):
        self.engagements += 1
        self.enabled = True
        self.last_command = self.clock.now
    def wait_until_ready(self):
        return True
    def require_profile(self, **expected):
        assert expected == dict(microsteps=32, baud=500000, protocol_version=3)
    def get_safety_faults(self):
        return self.fault
    def get_motor_status(self):
        if self.clock.now - self.last_command > .1:
            self.enabled = False
        return SimpleNamespace(motor_engaged=self.enabled)
    def set_acceleration(self, value):
        self.last_command = self.clock.now
        self.accel = value
        self.commands.append((self.clock.now, value))
    def get_state(self):
        elapsed = self.clock.now - self.last_update
        if self.enabled and not self.stuck:
            self.arm += self.velocity*elapsed + .5*self.accel*elapsed**2
            self.velocity += self.accel*elapsed
        self.last_update = self.clock.now
        stamp = 1 if self.frozen else (2**32 - 500000 + round(self.clock.now*1e6)) % 2**32
        return State(stamp, self.arm, 0., self.velocity, 0.)


class HardwareTests(unittest.TestCase):
    def test_collection_runs_tare_and_three_trials(self):
        class Connection(Client):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.disengage_motor()
        clock = Clock()
        client = Connection(clock)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(wizard, "time", clock), patch("pendulum_lab.hardware.time", clock), \
                patch.object(wizard, "LowLevelClient", return_value=client), \
                patch("builtins.input", return_value="OFF"), \
                patch.object(wizard, "record_tare", return_value=.02) as tare, \
                patch.object(wizard, "record_free_swing") as free, \
                patch.object(wizard, "record_motor_sweep") as motor:
            path = Path(directory)/"session"
            args = parser().parse_args(["sysid", "collect", "--port", "fake", "--skip-motor", "--out-dir", str(path)])
            wizard.execute(args)
            tare.assert_called_once_with(client, path)
            self.assertEqual(free.call_count, 3)
            self.assertEqual(free.call_args.args[2:], (2, 3, 10., 100.))
            motor.assert_not_called()
            self.assertEqual(json.loads((path/"metadata.json").read_text())["collection_state"], "complete")

    def test_passive_stream_never_enables_motor_and_rejects_faults(self):
        clock = Clock()
        with patch.object(wizard, "time", clock):
            client = Client(clock)
            result = wizard._record_state_stream(client, 1, 100)
            self.assertEqual(len(result["t"]), 100)
            self.assertEqual(client.commands, [])
            self.assertEqual(client.engagements, 0)
            for client in (Client(clock, fault=2), Client(clock, frozen=True)):
                with self.assertRaises(RuntimeError):
                    wizard._record_state_stream(client, 1, 100)

    def test_motor_sweep_holds_with_heartbeat_and_requires_physical_confirmation(self):
        clock = Clock()
        client = Client(clock)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(wizard, "time", clock), patch("pendulum_lab.hardware.time", clock), \
                patch("builtins.input", side_effect=["OFF", "SWEEP", "YES"]):
            path = wizard.record_motor_sweep(client, Path(directory))
            with np.load(path) as data:
                self.assertEqual(set(np.round(data["target"], 4)), {0, round(math.pi/2, 4), round(-math.pi/2, 4)})
            self.assertFalse(client.enabled)
            self.assertEqual(client.engagements, 1)
            times = np.array(client.commands)[:, 0]
            self.assertLessEqual(np.max(np.diff(times)), .021)
            self.assertLessEqual(max(abs(a) for _, a in client.commands), 40.)
            self.assertTrue(json.loads((Path(directory)/"metadata.json").read_text())["physical_travel_confirmed"])

    def test_cancelled_and_timed_out_sweeps_never_silently_succeed(self):
        for stuck, answers, expected in [(False, ["OFF", "NO"], "cancelled"),
                                         (True, ["OFF", "SWEEP"], "timeout")]:
            clock = Clock()
            client = Client(clock, stuck=stuck)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(wizard, "time", clock), patch("pendulum_lab.hardware.time", clock), \
                    patch("builtins.input", side_effect=answers):
                with self.assertRaisesRegex(RuntimeError, expected):
                    wizard.record_motor_sweep(client, Path(directory))
                self.assertFalse(client.enabled)
                if not stuck:
                    self.assertEqual(client.engagements, 0)

    def test_fault_during_powered_sweep_disengages_and_preserves_failure(self):
        class Faulty(Client):
            def get_safety_faults(self):
                return 2 if len(self.commands) >= 5 else 0
        clock = Clock()
        client = Faulty(clock)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(wizard, "time", clock), patch("pendulum_lab.hardware.time", clock), \
                patch("builtins.input", side_effect=["OFF", "SWEEP"]):
            with self.assertRaisesRegex(RuntimeError, "fault"):
                wizard.record_motor_sweep(client, Path(directory))
            self.assertFalse(client.enabled)
            self.assertEqual(client.engagements, 1)
            self.assertEqual(len(client.commands), 5)
            self.assertEqual(json.loads((Path(directory)/"metadata.json").read_text())["motor_sweep_state"], "failed")


if __name__ == "__main__":
    unittest.main()
