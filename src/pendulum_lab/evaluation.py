"""Shared metrics: an upright pass-through does not count as balance."""
import math
import numpy as np
from .contracts import PROFILE

COLUMNS = ["time_s", "arm_rad", "theta_rad", "arm_vel_rad_s", "pen_vel_rad_s", "action"]


def summarize(rows, stop_reason, duration):
    x = np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))
    if len(x) == 0:
        return {"stop_reason": stop_reason, "samples": 0, "success": False}
    balanced = (abs(x[:, 2]) <= math.radians(15)) & (abs(x[:, 4]) <= 4)
    streak = longest = 0
    for value in balanced:
        streak = streak + 1 if value else 0
        longest = max(longest, streak)
    full = stop_reason == "duration" and len(x) == round(duration * PROFILE.hz)
    final_ticks = 10 * PROFILE.hz
    success = full and duration >= 15 and bool(balanced[-final_ticks:].all())
    return {"stop_reason": stop_reason, "samples": len(x), "success": success,
            "balanced_fraction": float(balanced.mean()),
            "longest_balance_s": longest / PROFILE.hz,
            "max_arm_deg": float(np.degrees(abs(x[:, 1])).max()),
            "final_theta_deg": float(np.degrees(x[-1, 2]))}
