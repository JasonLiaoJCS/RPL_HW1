"""Student implementation of SAC. See docs/assignment.md for the contract."""


class Controller:
    def __init__(self, *, params_path, config, checkpoint=None):
        self.params_path = params_path
        self.config = config
        # TODO: initialize your controller and optionally load your checkpoint.

    def reset(self):
        # TODO: clear per-episode controller state.
        pass

    def act(self, observation):
        """Return one normalized acceleration in [-1, 1] from a (24,) observation."""
        # TODO: implement SAC.
        raise NotImplementedError("Implement SAC in algorithms/sac.py")


def train(*, env_factory, config, output_dir):
    """Train in simulation and save your checkpoint(s) beneath output_dir."""
    # TODO: implement training, checkpointing and learning-curve logging.
    raise NotImplementedError("Implement SAC training")
