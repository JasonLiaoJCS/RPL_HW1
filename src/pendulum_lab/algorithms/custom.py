"""Optional free-form method. Keep the shared controller interface."""


class Controller:
    def __init__(self, *, params_path, config, checkpoint=None):
        self.params_path = params_path
        self.config = config
        # TODO: initialize your own method and optionally load a checkpoint.

    def reset(self):
        # TODO: clear any per-episode state.
        pass

    def act(self, observation):
        """Return one normalized acceleration in [-1, 1] from a (24,) observation."""
        raise NotImplementedError("Implement your own method in algorithms/custom.py")


def train(*, env_factory, config, output_dir):
    """Optional: implement this hook only if your method needs training."""
    raise NotImplementedError("Implement CUSTOM training if your method needs it")
