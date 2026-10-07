"""Shared simulation reward and configurable weights."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Quadratic penalties and optional bonuses; zero weights disable terms."""
    # The pendulum-angle penalty has unit weight.
    k_pen_vel: float = 0.001     # pendulum angular velocity penalty (small)
    k_motor_pos: float = 0.5     # motor position penalty (centring)
    k_motor_vel: float = 0.005   # motor angular velocity penalty (small)
    k_action: float = 0.20       # action magnitude penalty

    # Optional changes in commanded action and observed motor velocity.
    k_action_rate: float = 0.0   # penalty on (a_t - a_{t-1})² ("command jerk")
    k_motor_jerk: float = 0.0    # penalty on (motor_vel_t - motor_vel_{t-1})²

    # Smooth bonus for small upright error and low motor speed.
    k_stillness_bonus: float = 0.0
    sigma_theta: float = 0.3              # bonus active within ~17°
    sigma_motor_vel: float = 1.0          # full bonus only at α̇ < ~1 rad/s

    # Per-step offset and a bonus gated by upright error and pendulum speed.
    k_alive_offset: float = 0.0
    k_upright_alive: float = 0.0
    alive_theta_band: float = 0.2618      # 15°
    alive_pen_vel_limit: float = 2.0      # rad/s

    # Optional pendulum overspeed penalty near upright.
    k_upright_overspeed: float = 0.0
    overspeed_sigma_theta: float = math.radians(30.0)
    overspeed_pen_vel_limit: float = 2.0


def compute_reward(
    *,
    theta: float,
    pen_vel: float,
    motor_pos: float,
    motor_vel: float,
    action: float,
    prev_action: float,
    prev_motor_vel: float,
    weights: RewardWeights,
) -> float:
    """Reward = alive terms + stillness bonus - quadratic cost - overspeed penalty.

    Sign convention: theta = 0 is upright (after wrapping), action ∈ [-1, 1],
    motor_vel and pen_vel are SI angular velocity (rad/s). The pendulum-angle
    quadratic term has implicit weight 1.0 — `RewardWeights` weights everything
    else relative to it.
    """
    action_delta = action - prev_action
    motor_vel_delta = motor_vel - prev_motor_vel
    cost = (
        theta * theta
        + weights.k_pen_vel * pen_vel * pen_vel
        + weights.k_motor_pos * motor_pos * motor_pos
        + weights.k_motor_vel * motor_vel * motor_vel
        + weights.k_action * action * action
        + weights.k_action_rate * action_delta * action_delta
        + weights.k_motor_jerk * motor_vel_delta * motor_vel_delta
    )
    if weights.k_stillness_bonus > 0.0:
        sigma_theta_sq = weights.sigma_theta * weights.sigma_theta
        sigma_motor_vel_sq = weights.sigma_motor_vel * weights.sigma_motor_vel
        upright_score = math.exp(-(theta * theta) / sigma_theta_sq)
        stillness_score = math.exp(-(motor_vel * motor_vel) / sigma_motor_vel_sq)
        bonus = weights.k_stillness_bonus * upright_score * stillness_score
    else:
        bonus = 0.0
    alive = weights.k_alive_offset
    if (weights.k_upright_alive > 0.0
            and abs(theta) <= weights.alive_theta_band
            and abs(pen_vel) <= weights.alive_pen_vel_limit):
        alive += weights.k_upright_alive
    overspeed_penalty = 0.0
    if weights.k_upright_overspeed > 0.0:
        excess_speed = max(abs(pen_vel) - weights.overspeed_pen_vel_limit, 0.0)
        upright_gate = math.exp(
            -(theta * theta)
            / (weights.overspeed_sigma_theta * weights.overspeed_sigma_theta)
        )
        overspeed_penalty = (
            weights.k_upright_overspeed * upright_gate * excess_speed * excess_speed
        )
    return float(alive + bonus - cost - overspeed_penalty)
