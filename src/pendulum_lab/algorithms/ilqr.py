"""Student implementation of ILQR. See docs/assignment.md for the contract."""


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
        # TODO: implement ILQR.
        raise NotImplementedError("Implement ILQR in algorithms/ilqr.py")
