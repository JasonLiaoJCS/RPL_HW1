import unittest

import numpy as np

from pendulum_lab.contracts import PARAMS, PROFILE
from pendulum_lab.environment import make_env


class NominalPlantTests(unittest.TestCase):
    def test_fixed_model_settings_across_initial_states(self):
        env = make_env(PARAMS, start="upright", duration=1)
        try:
            plant = env.unwrapped
            expected = None
            positions = []
            for seed in (1, 2, 3):
                obs, _ = env.reset(seed=seed)
                values = np.concatenate([
                    plant.model.body_mass.ravel(), plant.model.body_inertia.ravel(),
                    plant.model.body_ipos.ravel(), plant.model.dof_damping,
                    plant.model.dof_frictionloss, plant.model.opt.gravity,
                    [plant._transport_delay_s, plant._planner_speed_limit_rad_s,
                     plant._obs_staleness_s, plant._motor_max_accel_rad_s2]])
                if expected is None:
                    expected = values.copy()
                np.testing.assert_array_equal(values, expected)
                self.assertEqual(obs.shape, (24,))
                positions.append(plant.data.qpos.copy())
                before = plant.data.time
                env.step(np.zeros(1, np.float32))
                self.assertAlmostEqual(plant.data.time - before, PROFILE.dt)
            self.assertFalse(np.array_equal(positions[0], positions[1]))
        finally:
            env.close()

    def test_seeded_trajectory_repeats(self):
        trajectories = []
        for _ in range(2):
            env = make_env(PARAMS, start="mixed", duration=.5)
            try:
                obs, _ = env.reset(seed=42)
                rows = [obs.copy()]
                for tick in range(25):
                    obs, _, ended, done, _ = env.step(np.array([.1*np.sin(tick)]))
                    rows.append(obs.copy())
                    if ended or done:
                        break
                trajectories.append(rows)
            finally:
                env.close()
        np.testing.assert_array_equal(trajectories[0], trajectories[1])


if __name__ == "__main__":
    unittest.main()
