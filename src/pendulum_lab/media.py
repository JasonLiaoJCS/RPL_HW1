"""Offscreen simulator GIFs for side-by-side comparison with hardware video."""
from .contracts import PROFILE


class GifRecorder:
    """Render every `every`-th control tick from a fixed camera."""

    def __init__(self, env, *, width=480, height=360, every=2):
        import mujoco
        pen = env.unwrapped
        self.data = pen.data
        self.renderer = mujoco.Renderer(pen.model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.0, 0.0, 0.0)
        self.camera.distance = 0.42
        self.camera.azimuth = 135.0
        self.camera.elevation = -15.0
        self.every = every
        self.ticks = 0
        self.frames = []

    def capture(self):
        if self.ticks % self.every == 0:
            self.renderer.update_scene(self.data, self.camera)
            self.frames.append(self.renderer.render().copy())
        self.ticks += 1

    def save(self, path):
        from PIL import Image
        try:
            images = [Image.fromarray(frame) for frame in self.frames]
            if images:
                images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                               duration=round(1000 * self.every * PROFILE.dt))
        finally:
            self.renderer.close()
