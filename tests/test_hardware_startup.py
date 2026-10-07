"""Regression coverage for cached firmware feedback at hardware startup."""
import unittest
from unittest.mock import patch

from pendulum_lab.contracts import PARAMS
from pendulum_lab.controllers import load_controller
from pendulum_lab.hardware import run
from pendulum_lab.serial_client import State
from test_infrastructure import Clock, Client


class CachedFeedbackTests(unittest.TestCase):
    def exercise(self, *, frozen=False):
        clock = Clock()

        class CachedClient(Client):
            def get_state(self):
                if not (frozen and self.engaged):
                    self.timestamp = (round(clock.now * 1_000_000)//2000)*2000
                return State(self.timestamp, 0, 0, 0, 0)

        client = CachedClient()
        rows = []
        with patch("pendulum_lab.hardware.time", clock):
            try:
                run(load_controller("zero", PARAMS), "fake", .1, rows,
                    client_factory=lambda *a, **kw: client, confirm=lambda _: "RUN")
            finally:
                self.assertFalse(client.engaged)
                self.assertTrue(client.closed)
                if frozen:
                    self.assertEqual(client.commands, [])
        return rows, client

    def test_first_control_read_waits_for_new_firmware_sample(self):
        rows, client = self.exercise()
        self.assertEqual(len(rows), 5)
        self.assertEqual(client.commands, [0]*5)
        self.assertAlmostEqual(rows[0][0], .02)
        self.assertAlmostEqual(rows[-1][0], .1)

    def test_truly_frozen_feedback_still_aborts_before_command(self):
        with self.assertRaisesRegex(RuntimeError, r"Stale.*delta_us=0"):
            self.exercise(frozen=True)


if __name__ == "__main__":
    unittest.main()
