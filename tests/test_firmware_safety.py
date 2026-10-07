"""Firmware-boundary mathematics and host fault/status regression tests."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from pendulum_lab.hardware import validate_state
from pendulum_lab.serial_client import LowLevelClient, State
from test_infrastructure import Client
import test_infrastructure


class SafetyTests(unittest.TestCase):
    def test_firmware_safety_limits_native(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ required for native firmware-boundary tests")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "limits"
            subprocess.run([compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror",
                            "-I", str(root / "firmware/LowLevelServer"),
                            str(root / "tests/firmware_safety_limits.cpp"),
                            "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)

    def test_encoder_speed_spike_is_rejected(self):
        validate_state(State(100, 0, 0, 0, 50))
        for speed in (-83, 83):
            with self.assertRaisesRegex(RuntimeError, "Pendulum speed"):
                validate_state(State(100, 0, 0, 0, speed))

    def test_latched_fault_prevents_motor_enable(self):
        client = Client()
        client.get_safety_faults = lambda: 2
        with self.assertRaisesRegex(RuntimeError, "safety fault latched"):
            test_infrastructure.HardwareTests().exercise(client)
        self.assertFalse(client.ever_engaged)
        self.assertEqual(client.commands, [])

    def test_safety_status_parser(self):
        class Serial:
            def write(self, value):
                self.command = value
            def flush(self):
                pass
            def read(self, count):
                return self.response
        client = LowLevelClient("fake")
        client._ser = Serial()
        for flags in (0, 1, 2, 4, 8, 15):
            client._ser.response = bytes([13, 1, flags])
            self.assertEqual(client.get_safety_faults(), flags)
            self.assertEqual(client._ser.command, bytes([13]))
        for value in (b"", bytes([12, 1, 0]), bytes([13, 2, 0]), bytes([13, 1, 16])):
            client._ser.response = value
            with self.assertRaisesRegex(RuntimeError, "safety-status"):
                client.get_safety_faults()


if __name__ == "__main__":
    unittest.main()
