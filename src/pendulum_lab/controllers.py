"""The runner loads student code through this small interface."""
import importlib
import json

ALGORITHMS = ("zero", "pid", "lqr", "ilqr", "mpc", "ppo", "sac", "dqn", "custom")
RL = ("ppo", "sac", "dqn")
TRAINABLE = (*RL, "custom")


class Zero:
    """Infrastructure smoke test, not a balancing controller."""
    def reset(self):
        pass

    def act(self, observation):
        return 0.0


def algorithm_module(algorithm):
    if algorithm not in ALGORITHMS[1:]:
        raise ValueError("Unknown student algorithm")
    return importlib.import_module(f"pendulum_lab.algorithms.{algorithm}")


def load_controller(algorithm, params, config_path=None, checkpoint=None):
    if algorithm == "zero":
        return Zero()
    config = json.loads(config_path.read_text()) if config_path else {}
    return algorithm_module(algorithm).Controller(
        params_path=params, config=config, checkpoint=checkpoint)
