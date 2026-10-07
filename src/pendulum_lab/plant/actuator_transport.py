"""Kinematic model of FastAccelStepper's queued acceleration path.

The host sends piecewise-constant acceleration commands, but pulses already
queued by FastAccelStepper cannot be edited. A new command therefore becomes
active after a short transport delay. This module deliberately has no MuJoCo
or Gym dependency so the actuator timing can be unit-tested in isolation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class DelayedAccelerationPlanner:
    """Integrate delayed acceleration with a symmetric physical speed rail."""

    delay_s: float
    speed_limit_rad_s: float
    time_s: float = 0.0
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    active_accel_rad_s2: float = 0.0
    _queue: deque[tuple[float, float]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.configure(self.delay_s, self.speed_limit_rad_s)

    def configure(self, delay_s: float, speed_limit_rad_s: float) -> None:
        if delay_s < 0.0:
            raise ValueError(f"delay_s must be >= 0, got {delay_s}")
        if speed_limit_rad_s <= 0.0:
            raise ValueError(
                "speed_limit_rad_s must be > 0, "
                f"got {speed_limit_rad_s}"
            )
        self.delay_s = float(delay_s)
        self.speed_limit_rad_s = float(speed_limit_rad_s)

    def reset(
        self,
        *,
        position_rad: float = 0.0,
        velocity_rad_s: float = 0.0,
        active_accel_rad_s2: float = 0.0,
    ) -> None:
        self.time_s = 0.0
        self.position_rad = float(position_rad)
        limit = self.speed_limit_rad_s
        self.velocity_rad_s = max(-limit, min(limit, float(velocity_rad_s)))
        self.active_accel_rad_s2 = float(active_accel_rad_s2)
        self._queue.clear()

    def command(self, accel_rad_s2: float) -> None:
        """Queue a new acceleration command at the current planner time."""
        self._queue.append(
            (self.time_s + self.delay_s, float(accel_rad_s2))
        )
        self._activate_due_commands()

    def advance(self, dt_s: float) -> tuple[float, float]:
        """Advance and return ``(position, mean_actual_acceleration)``.

        The returned acceleration accounts for a command becoming active
        partway through the interval and time spent saturated at the speed
        rail. It is the acceleration that should couple into the pendulum.
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")

        end_time = self.time_s + float(dt_s)
        accel_impulse = 0.0
        while self._queue and self._queue[0][0] < end_time - 1e-15:
            activation_time, new_accel = self._queue.popleft()
            segment_s = max(0.0, activation_time - self.time_s)
            accel_impulse += self._integrate_segment(segment_s)
            self.time_s = max(self.time_s, activation_time)
            self.active_accel_rad_s2 = new_accel

        remaining_s = end_time - self.time_s
        accel_impulse += self._integrate_segment(remaining_s)
        self.time_s = end_time
        self._activate_due_commands()
        return self.position_rad, accel_impulse / float(dt_s)

    def clamp_position(self, lower_rad: float, upper_rad: float) -> None:
        """Clamp position and cancel velocity that points farther outward."""
        if self.position_rad >= upper_rad:
            self.position_rad = float(upper_rad)
            if self.velocity_rad_s > 0.0:
                self.velocity_rad_s = 0.0
        elif self.position_rad <= lower_rad:
            self.position_rad = float(lower_rad)
            if self.velocity_rad_s < 0.0:
                self.velocity_rad_s = 0.0

    def mirror(self) -> None:
        """Reflect all signed actuator state for symmetry augmentation."""
        self.position_rad = -self.position_rad
        self.velocity_rad_s = -self.velocity_rad_s
        self.active_accel_rad_s2 = -self.active_accel_rad_s2
        self._queue = deque((t, -a) for t, a in self._queue)

    def _activate_due_commands(self) -> None:
        while self._queue and self._queue[0][0] <= self.time_s + 1e-15:
            _, self.active_accel_rad_s2 = self._queue.popleft()

    def _integrate_segment(self, dt_s: float) -> float:
        if dt_s <= 0.0:
            return 0.0

        accel = self.active_accel_rad_s2
        velocity = self.velocity_rad_s
        limit = self.speed_limit_rad_s
        if accel == 0.0:
            self.position_rad += velocity * dt_s
            return 0.0

        target_limit = limit if accel > 0.0 else -limit
        time_to_limit = (target_limit - velocity) / accel
        if 0.0 < time_to_limit < dt_s:
            self.position_rad += (
                velocity * time_to_limit
                + 0.5 * accel * time_to_limit * time_to_limit
                + target_limit * (dt_s - time_to_limit)
            )
            self.velocity_rad_s = target_limit
            return accel * time_to_limit

        new_velocity = velocity + accel * dt_s
        if accel > 0.0:
            new_velocity = min(new_velocity, limit)
        else:
            new_velocity = max(new_velocity, -limit)
        actual_accel = (new_velocity - velocity) / dt_s
        self.position_rad += velocity * dt_s + 0.5 * actual_accel * dt_s * dt_s
        self.velocity_rad_s = new_velocity
        return actual_accel * dt_s
