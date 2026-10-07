"""Gymnasium environment for the rotary inverted pendulum, parameterised
from a per-rig JSON parameter file.

Geometry (Furuta pendulum):
    - Arm rotates about a vertical axis, driven by a position-controlled
      "motor" (modelling the stepper and driver as a stiff PD actuator).
    - Pendulum hinges at the end of the arm, free-swinging in the vertical
      plane perpendicular to the arm direction. Hangs straight down at rest.

Observation:
    Each default frame is [motor_pos, sin(theta), cos(theta), motor_vel,
    pendulum_vel, previous_action]. Four frames form the lab's 24-value
    observation; theta = 0 is upright.

Action (1-dim, in [-1, 1]) — interpreted per `action_mode`:
    - "accel" (default): commanded angular acceleration in
      [-max_accel, +max_accel] rad/s²; integrated to a capped velocity and
      then a position target fed to the PD actuator.
    - "velocity": commanded angular velocity in [-max_vel, +max_vel] rad/s.
      Converted to an acceleration each tick via a saturating P-law
      (accel = clip((v_des − v)/dt, ±max_accel)) and then integrated exactly
      like accel mode — same firmware transport (CMD_SET_ACCEL), one fewer
      integrator between action and pendulum coupling. The velocity cap
      becomes the explicit action bound instead of a hidden saturation.
    - "position_delta": per-step motor-position delta
      (action × max_action_delta_rad), integrated into the commanded target,
      clamped, and fed to the PD actuator through a first-order
      motor-bandwidth lag.

Reward:
    Quadratic state/action penalties and optional bonuses; see reward.py.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .actuator_transport import DelayedAccelerationPlanner
from .pendulum_geometry import (
    PENDULUM_COM_M,
    PENDULUM_I_COM_SWING_KG_M2,
    PENDULUM_MASS_KG,
)
from .reward import RewardWeights, compute_reward


HERE = Path(__file__).resolve().parent
# Uncalibrated starter parameters; pass your identified file through --params.
DEFAULT_PARAMS_PATH = HERE / "data" / "example_rig.json"

# Internal model limits. The lab wrapper enforces the narrower +/-100-degree envelope.
MOTOR_LIMIT_RAD = math.radians(135.0)
MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)

# Nominal arm geometry for the single-bearing assembly; validate for your kit.
ARM_LENGTH_M = 0.065
ARM_MASS_KG = 0.030
ARM_COM_M = 0.035

GRAVITY = 9.81

# AS5600 encoder resolution.
PENDULUM_LSB_RAD = 2.0 * math.pi / 4096.0

# Firmware measurement model (see LowLevelServer.ino). The firmware keeps a
# 500 Hz ring buffer of encoder samples; GET_STATE returns positions from
# the newest sample and velocities as (newest − oldest)/Δt over a 5-sample
# (8 ms) window. Motor position is the step counter; the pendulum is the
# 12-bit AS5600. Quantised finite differences over the window reproduce the
# velocity quantization: one pendulum LSB over the window is about 0.19 rad/s;
# one motor step at 1/32 is about 0.12 rad/s. Enabled via `firmware_obs_model`;
# when on it feeds both the
# observation AND the velocity-mode P-law (the real host uses the same
# GET_STATE read for both).
#
# MOTOR_MICROSTEPS must match the firmware and the driver's physical settings.
# The lab uses DRV8825 1/32 microstepping with MODE0, MODE1 and MODE2 high.
MOTOR_MICROSTEPS = 32
MOTOR_STEP_LSB_RAD = 2.0 * math.pi / (200.0 * MOTOR_MICROSTEPS)
FIRMWARE_VEL_WINDOW_S = 0.008
# Fixed age of the firmware state snapshot when the host receives it.
OBS_STALENESS_NOMINAL_S = 0.004

# Nominal actuator settings; the lab wrapper supplies its control contract.
MAX_ACTION_DELTA_RAD = 0.10
CONTROL_FREQ_HZ = 50.0
MAX_VELOCITY_RAD_S = 3.5
MOTOR_PLANNER_SPEED_LIMIT_RAD_S = 5.0
TRANSPORT_DELAY_NOMINAL_S = 0.010
MAX_ACCEL_RAD_S2 = 150.0
UPSTREAM_TIP_MASS_KG = 0.00712


@dataclass(frozen=True)
class PendulumParams:
    pendulum_mass_kg: float
    pendulum_com_m: float       # perpendicular distance, joint axis -> COM
    pendulum_inertia_kg_m2: float  # about the joint axis (m·d² + I_com_swing)
    pendulum_friction: float    # viscous, N·m·s
    pendulum_coulomb: float     # Coulomb (dry) friction torque, N·m

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PendulumParams":
        """Construct from a per-rig sysid document.

        The URDF remains the fallback geometry, but a rig may replace the
        nominal UK 2p tip mass or provide a directly measured whole-pendulum
        mass. Free-swing sysid is also allowed to override pivot inertia.
        """
        path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
        with open(path) as f:
            doc = json.load(f)
        pen = doc["pendulum"]["derived"]
        run_meta = doc.get("run_meta", {})
        com = float(run_meta.get("pendulum_com_m", PENDULUM_COM_M))
        tip = run_meta.get("pendulum_tip_mass", {})
        if "pendulum_total_mass_kg" in run_meta:
            mass = float(run_meta["pendulum_total_mass_kg"])
        elif tip.get("coin") == "New Taiwan dollar NT$10":
            mass = (
                PENDULUM_MASS_KG
                + float(tip["mass_kg"])
                - UPSTREAM_TIP_MASS_KG
            )
        else:
            mass = PENDULUM_MASS_KG
        pivot_inertia = float(
            pen.get(
                "inertia_measured_kg_m2",
                mass * com**2 + PENDULUM_I_COM_SWING_KG_M2,
            )
        )
        viscous = float(pen["viscous_friction_N_m_s"])
        coulomb = float(pen.get("coulomb_friction_N_m", 0.0))
        if (not np.isfinite([mass, com, pivot_inertia, viscous, coulomb]).all()
                or mass <= 0.0 or com <= 0.0 or pivot_inertia <= mass * com**2
                or viscous < 0.0 or coulomb < 0.0):
            raise ValueError(
                "per-rig mass/inertia imply non-positive COM inertia: "
                f"mass={mass}, pivot_inertia={pivot_inertia}"
            )
        return cls(
            pendulum_mass_kg=mass,
            pendulum_com_m=com,
            pendulum_inertia_kg_m2=pivot_inertia,
            pendulum_friction=viscous,
            pendulum_coulomb=coulomb,
        )


def build_mjcf(p: PendulumParams) -> str:
    """Construct an MJCF model string parameterised by sysid params.

    Pendulum inertia decomposition: MJCF expects the inertia tensor in the
    body's *inertial* frame, expressed about the COM. Our pendulum body's
    frame x-axis is aligned with the joint's rotation axis (axis="1 0 0").
    For rotation about the joint axis through the joint origin, the
    parallel-axis theorem gives
        I_about_joint_x = I_com_xx + m * (perpendicular distance from x-axis to COM)^2
                       = I_com_xx + m * (y_com^2 + z_com^2)
    With COM at (0, 0, -d), perpendicular distance is d.
    We back-compute I_com_xx = I_axis − m·d² from the selected per-rig pivot
    inertia. This is essential when the physical tip mass differs from the
    URDF. For the off-axis components: a rod-like pendulum extending along z
    has appreciable extent
    perpendicular to y, so iyy ≈ ixx; izz (length-axis) is near zero.
    Slight asymmetry doesn't affect 1-DOF swing dynamics but keeps MuJoCo's
    solver well-conditioned.

    The motor is approximated by a stiff position actuator with damping
    based on effective joint inertia. This assumes the physical motor
    follows the commanded steps; it does not model missed steps.
    """
    m = p.pendulum_mass_kg
    d = p.pendulum_com_m
    I_com_swing = max(1e-7, p.pendulum_inertia_kg_m2 - m * d**2)
    # diaginertia[0] = ixx = swing-axis inertia about COM
    # diaginertia[1] = iyy ≈ ixx for a rod-like body with extent perpendicular to its length axis
    # diaginertia[2] = izz ≈ tiny (length-axis inertia)
    diag_inertia = (I_com_swing, I_com_swing, 1e-7)

    # Approximate arm COM inertia with the thin-rod formula m*L²/12.
    arm_I = ARM_MASS_KG * ARM_LENGTH_M ** 2 / 12.0

    # PD position-actuator gains. The motor joint sees the effective
    # inertia (arm parallel-axis + pendulum mass at arm tip + pendulum
    # self-inertia about its own joint), not just arm_I; kv is critical
    # damping against that inertia.
    I_arm_about_motor = arm_I + ARM_MASS_KG * ARM_COM_M ** 2
    I_pen_at_arm_tip  = p.pendulum_mass_kg * ARM_LENGTH_M ** 2
    I_pen_self        = p.pendulum_inertia_kg_m2
    I_motor_joint     = I_arm_about_motor + I_pen_at_arm_tip + I_pen_self
    kp = 1000.0
    kv = 2.0 * math.sqrt(kp * I_motor_joint)  # critical damping

    return f"""<?xml version="1.0"?>
<mujoco model="rotary_inverted_pendulum">
  <option timestep="0.001" gravity="0 0 -{GRAVITY}" integrator="implicitfast"/>

  <default>
    <joint armature="0" damping="0"/>
    <geom contype="0" conaffinity="0" rgba="0.6 0.6 0.6 1"/>
  </default>

  <worldbody>
    <light diffuse="0.7 0.7 0.7" pos="0 0 1" dir="0 0 -1"/>

    <body name="arm" pos="0 0 0">
      <joint name="motor_joint" type="hinge" axis="0 0 1"
             range="-{MOTOR_LIMIT_RAD} {MOTOR_LIMIT_RAD}" limited="true"
             damping="0.0001" frictionloss="0"/>
      <inertial pos="{ARM_COM_M} 0 0" mass="{ARM_MASS_KG}"
                diaginertia="1e-6 {arm_I} {arm_I}"/>
      <geom name="arm_visual" type="capsule"
            fromto="0 0 0 {ARM_LENGTH_M} 0 0" size="0.004"
            rgba="0.4 0.6 0.8 1"/>

      <body name="pendulum" pos="{ARM_LENGTH_M} 0 0">
        <joint name="pendulum_joint" type="hinge" axis="1 0 0"
               damping="{p.pendulum_friction}"
               frictionloss="{p.pendulum_coulomb}"/>
        <inertial pos="0 0 -{d}" mass="{m}"
                  diaginertia="{diag_inertia[0]} {diag_inertia[1]} {diag_inertia[2]}"/>
        <geom name="pendulum_visual" type="capsule"
              fromto="0 0 0 0 0 -{d * 2}" size="0.003"
              rgba="0.8 0.4 0.4 1"/>
        <geom name="pendulum_tip" type="sphere" pos="0 0 -{d * 2}"
              size="0.006" rgba="0.9 0.7 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="motor" joint="motor_joint"
              kp="{kp}" kv="{kv}"
              ctrlrange="-{MOTOR_SAFE_LIMIT_RAD} {MOTOR_SAFE_LIMIT_RAD}"/>
  </actuator>
</mujoco>
"""


class RotaryInvertedPendulumEnv(gym.Env):
    """Off-board MuJoCo simulation of the rotary inverted pendulum.

    The MuJoCo pendulum joint angle phi is zero hanging and pi upright.
    Observations use theta = wrap(phi - pi), so theta is zero upright.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        params_path: str | Path | None = None,
        control_freq_hz: float = CONTROL_FREQ_HZ,
        action_mode: str = "accel",  # The lab wrapper requires acceleration actions.
        max_accel_rad_s2: float = MAX_ACCEL_RAD_S2,  # accel-mode: action × this = commanded angular accel
        max_velocity_rad_s: float = MAX_VELOCITY_RAD_S,  # accel-mode: velocity saturation cap
        max_action_delta_rad: float = MAX_ACTION_DELTA_RAD,  # position-mode: action × this = per-step target delta
        episode_length_s: float = 8.0,
        # Fraction of initial states near upright rather than hanging.
        upright_reset_frac: float = 0.0,
        reset_arm_limit_rad: float = math.radians(5.0),
        # Optional reset bank; each row is one stacked observation.
        spin_reset_bank: str | Path | None = None,
        spin_reset_frac: float = 0.0,
        # Optional initial-state ranges for recovery trials.
        recovery_reset_frac: float = 0.0,
        recovery_theta_max_rad: float = math.radians(30.0),
        recovery_pen_vel_min_rad_s: float = 2.0,
        recovery_pen_vel_max_rad_s: float = 8.0,
        recovery_motor_vel_max_rad_s: float = 5.0,
        # Fraction of resets with the arm centered and pendulum hanging still.
        exact_hanging_reset_frac: float = 0.0,
        # Frames are oldest to newest; reset repeats the initial frame.
        obs_history_len: int = 4,
        # The lab uses velocity channels. Position-only observations require at least two frames.
        obs_include_velocities: bool = True,
        # Quantise positions and estimate velocities over an 8 ms window,
        # with snapshot age set by obs_staleness_s. Matching observation
        # shapes alone does not make checkpoints compatible across settings.
        firmware_obs_model: bool = False,
        obs_staleness_s: float | None = None,  # None → OBS_STALENESS_NOMINAL_S
        # Quadratic penalty weights and optional bonuses; see reward.py.
        reward_motor_pos_weight: float = 0.5,
        reward_motor_vel_weight: float = 0.005,
        reward_action_weight: float = 0.20,
        reward_pen_vel_weight: float = 0.001,
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,        # bonus active within ~17°
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,  # full bonus only at α̇ < ~1 rad/s
        # None uses the RewardWeights default for each optional term.
        reward_alive_offset: float | None = None,
        reward_upright_alive_weight: float | None = None,
        reward_upright_overspeed_weight: float | None = None,
        reward_motor_jerk_weight: float | None = None,
        reward_action_rate_weight: float | None = None,
        render_mode: str | None = None,
        # Fixed actuator and transport settings.
        motor_max_accel_rad_s2: float | None = None,  # None => use max_accel_rad_s2
        action_delay_steps: int = 0,
        action_lag_tau_s: float = 0.0,
        # Pure delay between the host issuing CMD_SET_ACCEL and the queued
        # FastAccelStepper trajectory changing. The physical planner has its
        # own speed rail, distinct from the policy's ±max_velocity request.
        transport_delay_s: float = 0.0,
        motor_planner_speed_limit_rad_s: float | None = None,
        # Moving average of the last N actions. The lab wrapper sets N=1 (disabled).
        action_smooth_window: int = 4,
        motor_tau_s: float = 0.0,  # position-mode fixed motor-bandwidth lag
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # Optional termination on accumulated pendulum travel. Disabled in
        # standard evaluation so failures remain observable for the full run.
        terminate_on_spin_loop: bool = False,
        spin_loop_revolutions: float = 3.0,
        spin_loop_penalty: float = 15.0,
        # Optional penalty for accumulated travel beyond the free allowance.
        spin_progress_penalty_weight: float = 0.0,
        spin_progress_free_revolutions: float = 1.5,
    ):
        super().__init__()
        if action_mode not in ("accel", "velocity", "position_delta"):
            raise ValueError(
                f"action_mode must be 'accel', 'velocity', or 'position_delta', "
                f"got {action_mode!r}"
            )
        self.params = PendulumParams.load(params_path)
        self.control_freq_hz = control_freq_hz
        self.action_mode = action_mode
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.max_velocity_rad_s = max_velocity_rad_s
        self.max_action_delta_rad = max_action_delta_rad
        self.episode_length_s = episode_length_s
        if not 0.0 <= upright_reset_frac <= 1.0:
            raise ValueError(f"upright_reset_frac must be in [0, 1], got {upright_reset_frac}")
        self.upright_reset_frac = float(upright_reset_frac)
        if not 0 <= reset_arm_limit_rad < MOTOR_SAFE_LIMIT_RAD:
            raise ValueError("Invalid initial arm envelope")
        self.reset_arm_limit_rad = float(reset_arm_limit_rad)
        if not 0.0 <= spin_reset_frac <= 1.0:
            raise ValueError(f"spin_reset_frac must be in [0, 1], got {spin_reset_frac}")
        if not 0.0 <= recovery_reset_frac <= 1.0:
            raise ValueError(
                "recovery_reset_frac must be in [0, 1], got "
                f"{recovery_reset_frac}"
            )
        if not 0.0 <= exact_hanging_reset_frac <= 1.0:
            raise ValueError(
                "exact_hanging_reset_frac must be in [0, 1], got "
                f"{exact_hanging_reset_frac}"
            )
        if (upright_reset_frac + spin_reset_frac + recovery_reset_frac
                + exact_hanging_reset_frac > 1.0):
            raise ValueError(
                "upright_reset_frac + spin_reset_frac + recovery_reset_frac + "
                "exact_hanging_reset_frac must be <= 1"
            )
        self.spin_reset_frac = float(spin_reset_frac)
        self.recovery_reset_frac = float(recovery_reset_frac)
        self.exact_hanging_reset_frac = float(exact_hanging_reset_frac)
        if recovery_theta_max_rad <= 0.0 or recovery_theta_max_rad > math.pi:
            raise ValueError("recovery_theta_max_rad must be in (0, pi]")
        if (recovery_pen_vel_min_rad_s < 0.0
                or recovery_pen_vel_max_rad_s
                < recovery_pen_vel_min_rad_s):
            raise ValueError(
                "recovery pendulum-speed bounds must satisfy 0 <= min <= max"
            )
        if recovery_motor_vel_max_rad_s < 0.0:
            raise ValueError("recovery_motor_vel_max_rad_s must be >= 0")
        self.recovery_theta_max_rad = float(recovery_theta_max_rad)
        self.recovery_pen_vel_min_rad_s = float(recovery_pen_vel_min_rad_s)
        self.recovery_pen_vel_max_rad_s = float(recovery_pen_vel_max_rad_s)
        self.recovery_motor_vel_max_rad_s = float(
            recovery_motor_vel_max_rad_s
        )
        if int(obs_history_len) < 1:
            raise ValueError(f"obs_history_len must be >= 1, got {obs_history_len}")
        self.obs_history_len = int(obs_history_len)
        self.obs_include_velocities = bool(obs_include_velocities)
        if not self.obs_include_velocities and self.obs_history_len < 2:
            raise ValueError(
                "obs_include_velocities=False needs obs_history_len >= 2 — "
                "velocities must be inferable from the position history"
            )
        self._obs_history: deque = deque(maxlen=self.obs_history_len)
        self.firmware_obs_model = bool(firmware_obs_model)
        self._fixed_obs_staleness_s = (
            float(obs_staleness_s) if obs_staleness_s is not None
            else OBS_STALENESS_NOMINAL_S
        )
        self._obs_staleness_s = self._fixed_obs_staleness_s
        # Ring of (motor_qpos, pendulum_qpos) captured every physics substep,
        # continuous across control steps within an episode. Sized to cover
        # the velocity window + worst-case staleness with margin.
        self._meas_ring: deque = deque(maxlen=32)
        self._prev_cmd_pos = 0.0
        self._meas_motor_pos = 0.0
        self._meas_pen_phi = 0.0
        self._meas_motor_vel = 0.0
        self._meas_pen_vel = 0.0
        # Reward terms and defaults are defined in reward.py.
        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_action_rate=(
                float(reward_action_rate_weight)
                if reward_action_rate_weight is not None else 0.0
            ),
            k_motor_jerk=(
                float(reward_motor_jerk_weight)
                if reward_motor_jerk_weight is not None else 0.0
            ),
            k_stillness_bonus=(
                float(reward_stillness_bonus_weight)
                if reward_stillness_bonus_weight is not None else 0.0
            ),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
            k_alive_offset=(
                float(reward_alive_offset)
                if reward_alive_offset is not None else 0.0
            ),
            k_upright_alive=(
                float(reward_upright_alive_weight)
                if reward_upright_alive_weight is not None else 0.0
            ),
            k_upright_overspeed=(
                float(reward_upright_overspeed_weight)
                if reward_upright_overspeed_weight is not None else 0.0
            ),
        )
        self._prev_action = 0.0
        self._prev_motor_vel = 0.0  # tracked for the motor-jerk reward term
        self.render_mode = render_mode

        # Fixed actuator configuration.
        self._fixed_motor_max_accel_rad_s2 = (
            float(motor_max_accel_rad_s2) if motor_max_accel_rad_s2 is not None
            else float(max_accel_rad_s2)
        )
        self._fixed_action_delay_steps = int(action_delay_steps)
        self._fixed_action_lag_tau_s = float(action_lag_tau_s)
        self._fixed_transport_delay_s = float(transport_delay_s)
        if self._fixed_transport_delay_s < 0.0:
            raise ValueError("transport_delay_s must be >= 0")
        self.motor_planner_speed_limit_rad_s = float(
            max_velocity_rad_s
            if motor_planner_speed_limit_rad_s is None
            else motor_planner_speed_limit_rad_s
        )
        if self.motor_planner_speed_limit_rad_s <= 0.0:
            raise ValueError("motor_planner_speed_limit_rad_s must be > 0")
        if int(action_smooth_window) < 1:
            raise ValueError(
                f"action_smooth_window must be >= 1, got {action_smooth_window}"
            )
        self.action_smooth_window = int(action_smooth_window)
        self._kick_ticks_left = 0
        self._kick_value = 0.0
        self._action_smooth_buf: deque = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )
        self._fixed_motor_tau_s = float(motor_tau_s)
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        if spin_loop_revolutions <= 0.0:
            raise ValueError("spin_loop_revolutions must be positive")
        self.terminate_on_spin_loop = bool(terminate_on_spin_loop)
        self.spin_loop_revolutions = float(spin_loop_revolutions)
        self.spin_loop_penalty = float(spin_loop_penalty)
        if spin_progress_penalty_weight < 0.0:
            raise ValueError("spin_progress_penalty_weight must be >= 0")
        if spin_progress_free_revolutions < 0.0:
            raise ValueError("spin_progress_free_revolutions must be >= 0")
        self.spin_progress_penalty_weight = float(
            spin_progress_penalty_weight
        )
        self.spin_progress_free_revolutions = float(
            spin_progress_free_revolutions
        )
        self._spin_travel_rad = 0.0
        self._spin_last_phi = 0.0
        self._spin_catch_ticks = 0
        self._spin_caught = False
        self._fixed_motor_frictionloss = 0.0  # set by user via reset(options=) if desired
        explicit_planner = (
            self._fixed_transport_delay_s > 0.0
            or abs(self.motor_planner_speed_limit_rad_s
                   - self.max_velocity_rad_s) > 1e-12
        )
        legacy_transport = (
            self._fixed_action_delay_steps > 0
            or self._fixed_action_lag_tau_s > 0.0
        )
        if explicit_planner and legacy_transport:
            raise ValueError(
                "explicit planner transport cannot be combined with "
                "action_delay_steps or action_lag_tau_s"
            )

        xml = build_mjcf(self.params)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Number of physics steps per control step.
        physics_dt = self.model.opt.timestep
        n_substeps = int(round((1.0 / control_freq_hz) / physics_dt))
        self._n_substeps = max(1, n_substeps)
        self._dt_control = self._n_substeps * physics_dt

        # Size the measurement ring for the velocity window plus worst-case
        # staleness at the actual physics dt (replaces the placeholder above).
        self._meas_win_substeps = max(1, int(round(FIRMWARE_VEL_WINDOW_S / physics_dt)))
        stale_max_s = self._fixed_obs_staleness_s
        stale_max_substeps = int(math.ceil(stale_max_s / physics_dt))
        self._meas_ring = deque(maxlen=self._meas_win_substeps + stale_max_substeps + 2)

        self._max_steps = int(episode_length_s * control_freq_hz)
        self._step_count = 0
        # Keep the host velocity request separate from the delayed planner state.
        self._host_velocity_cmd = 0.0
        self._motor_vel = 0.0      # delayed physical planner velocity, rad/s
        self._motor_target = 0.0   # integrated position target, rad
        self._motor_max_accel_rad_s2 = float(max_accel_rad_s2)
        self._transport_delay_s = 0.0
        self._planner_speed_limit_rad_s = self.motor_planner_speed_limit_rad_s
        self._accel_transport = DelayedAccelerationPlanner(
            delay_s=0.0,
            speed_limit_rad_s=self.motor_planner_speed_limit_rad_s,
        )
        # Position-delta mode: first-order lag on the commanded target that
        # models the stepper's ramp-to-target bandwidth. `_lagged_target` is
        # the filter state (what the PD actuator actually chases); `_motor_tau_s`
        # is the configured time constant.
        self._lagged_target = 0.0
        self._motor_tau_s = 0.0
        self._action_delay_steps = 0
        self._action_queue: deque = deque()
        # Continuous action lag: first-order LP filter on commanded action.
        # `_action_lag_tau_s` is set from the fixed configuration.
        # `_lagged_action` is the filter's internal state, reset each episode.
        self._action_lag_tau_s = 0.0
        self._lagged_action = 0.0
        # Velocity-mode transport lag state: the P-law runs host-side with
        # no delay, so lag/queue apply to the accel COMMAND it sends, not to
        # the policy's velocity setpoint.
        self._lagged_accel_cmd = 0.0
        self._accel_cmd_queue: deque = deque()

        # Cache joint addresses (faster than name lookup each step).
        self._motor_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        ]
        self._motor_qvel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        ]
        self._motor_dof_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        ]
        self._pen_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        ]
        self._pen_qvel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        ]
        self._pen_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum"
        )
        self._pen_dof_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        ]

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # Per-frame bounds include previous_action in [-1, 1]. Finite history
        # provides temporal context but does not expose all actuator state.
        frame_high = (
            np.array([MOTOR_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0], dtype=np.float32)
            if self.obs_include_velocities
            else np.array([MOTOR_LIMIT_RAD, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        obs_high = np.tile(frame_high, self.obs_history_len)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._spin_reset_obs: np.ndarray | None = None
        if spin_reset_bank is not None:
            if not self.obs_include_velocities:
                raise ValueError("spin reset banks currently require velocity observations")
            with np.load(Path(spin_reset_bank), allow_pickle=False) as bank:
                if "obs" not in bank:
                    raise ValueError("spin reset bank must contain an 'obs' array")
                reset_obs = np.asarray(bank["obs"], dtype=np.float32)
            if reset_obs.ndim != 2 or reset_obs.shape[1] != self.observation_space.shape[0]:
                raise ValueError(
                    "spin reset bank observation shape mismatch: "
                    f"got {reset_obs.shape}, expected (*, {self.observation_space.shape[0]})"
                )
            if len(reset_obs) == 0:
                raise ValueError("spin reset bank is empty")
            self._spin_reset_obs = reset_obs
        elif self.spin_reset_frac > 0.0:
            raise ValueError("spin_reset_frac > 0 requires spin_reset_bank")

        self._viewer = None

    # --- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._motor_max_accel_rad_s2 = self._fixed_motor_max_accel_rad_s2
        self._action_delay_steps = self._fixed_action_delay_steps
        self._action_lag_tau_s = self._fixed_action_lag_tau_s
        self._transport_delay_s = self._fixed_transport_delay_s
        self._planner_speed_limit_rad_s = self.motor_planner_speed_limit_rad_s
        self._motor_tau_s = self._fixed_motor_tau_s
        self._obs_staleness_s = self._fixed_obs_staleness_s
        self.model.opt.gravity[:] = (0.0, 0.0, -GRAVITY)
        self.model.dof_frictionloss[self._motor_dof_addr] = self._fixed_motor_frictionloss
        self.model.dof_damping[self._pen_dof_addr] = self.params.pendulum_friction
        self.model.dof_frictionloss[self._pen_dof_addr] = self.params.pendulum_coulomb

        self._action_queue = deque([0.0] * self._action_delay_steps,
                                   maxlen=max(1, self._action_delay_steps + 1))
        self._lagged_action = 0.0
        self._lagged_accel_cmd = 0.0
        self._accel_cmd_queue = deque([0.0] * self._action_delay_steps,
                                      maxlen=max(1, self._action_delay_steps + 1))
        self._kick_ticks_left = 0
        self._kick_value = 0.0
        self._action_smooth_buf = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )

        # Select the configured reset distribution. Joint phi=0 is hanging;
        # phi=pi is upright. The lab wrapper selects its prescribed starts.
        reset_draw = float(self.np_random.random())
        spin_obs = None
        if (self._spin_reset_obs is not None
                and reset_draw < self.spin_reset_frac):
            spin_obs = self._spin_reset_obs[
                int(self.np_random.integers(len(self._spin_reset_obs)))
            ]
            frames = spin_obs.reshape(self.obs_history_len, 6)
            newest = frames[-1]
            motor_pos0 = float(newest[0] + self.np_random.uniform(-0.05, 0.05))
            theta0 = math.atan2(float(newest[1]), float(newest[2]))
            theta0 += float(self.np_random.uniform(-math.radians(3), math.radians(3)))
            phi0 = math.pi + theta0
            motor_vel0 = float(newest[3] * self.np_random.uniform(0.9, 1.1))
            pen_vel0 = float(newest[4] * self.np_random.uniform(0.9, 1.1))
            prev_action0 = float(newest[5])
        elif (self.recovery_reset_frac > 0.0
                and reset_draw
                < self.spin_reset_frac + self.recovery_reset_frac):
            theta0 = float(self.np_random.uniform(
                -self.recovery_theta_max_rad,
                self.recovery_theta_max_rad,
            ))
            speed0 = float(self.np_random.uniform(
                self.recovery_pen_vel_min_rad_s,
                self.recovery_pen_vel_max_rad_s,
            ))
            pen_vel0 = speed0 * float(self.np_random.choice((-1.0, 1.0)))
            phi0 = math.pi + theta0
            # Sample arm position and velocity independently within these bounds.
            motor_pos0 = float(self.np_random.uniform(
                -0.5 * MOTOR_SAFE_LIMIT_RAD, 0.5 * MOTOR_SAFE_LIMIT_RAD
            ))
            motor_vel0 = float(self.np_random.uniform(
                -self.recovery_motor_vel_max_rad_s,
                self.recovery_motor_vel_max_rad_s,
            ))
            prev_action0 = float(np.clip(
                motor_vel0 / max(self.max_velocity_rad_s, 1e-9), -1.0, 1.0
            ))
        elif (self.upright_reset_frac > 0.0
                and reset_draw < self.spin_reset_frac
                + self.recovery_reset_frac + self.upright_reset_frac):
            phi0 = math.pi + self.np_random.uniform(-0.3, 0.3)
            pen_vel0 = self.np_random.uniform(-0.5, 0.5)
            motor_pos0 = float(self.np_random.uniform(
                -self.reset_arm_limit_rad, self.reset_arm_limit_rad
            ))
            motor_vel0 = 0.0
            prev_action0 = 0.0
        elif (self.exact_hanging_reset_frac > 0.0
                and reset_draw < self.spin_reset_frac
                + self.recovery_reset_frac + self.upright_reset_frac
                + self.exact_hanging_reset_frac):
            phi0 = 0.0
            pen_vel0 = 0.0
            motor_pos0 = 0.0
            motor_vel0 = 0.0
            prev_action0 = 0.0
        else:
            phi0 = self.np_random.uniform(-0.05, 0.05)
            pen_vel0 = 0.0
            motor_pos0 = float(self.np_random.uniform(
                -self.reset_arm_limit_rad, self.reset_arm_limit_rad
            ))
            motor_vel0 = 0.0
            prev_action0 = 0.0
        motor_pos0 = float(np.clip(
            motor_pos0, -0.9 * MOTOR_SAFE_LIMIT_RAD, 0.9 * MOTOR_SAFE_LIMIT_RAD
        ))
        self.data.qpos[self._motor_qpos_addr] = motor_pos0
        self.data.qpos[self._pen_qpos_addr] = phi0
        self.data.qvel[self._motor_qvel_addr] = motor_vel0
        self.data.qvel[self._pen_qvel_addr] = pen_vel0
        self._motor_target = float(self.data.qpos[self._motor_qpos_addr])
        self._lagged_target = self._motor_target  # start the lag at the true start pos
        self._prev_cmd_pos = self._motor_target   # substep interpolation anchor
        self._motor_vel = motor_vel0
        self._host_velocity_cmd = float(np.clip(
            prev_action0 * self.max_velocity_rad_s,
            -self.max_velocity_rad_s,
            self.max_velocity_rad_s,
        ))
        self._accel_transport.configure(
            self._transport_delay_s,
            self._planner_speed_limit_rad_s,
        )
        self._accel_transport.reset(
            position_rad=self._motor_target,
            velocity_rad_s=motor_vel0,
        )
        self._step_count = 0
        self._prev_action = prev_action0
        self._prev_motor_vel = motor_vel0
        self._spin_travel_rad = 0.0
        self._spin_last_phi = float(self.data.qpos[self._pen_qpos_addr])
        self._spin_catch_ticks = 0
        self._spin_caught = False
        self._obs_history.clear()
        if spin_obs is not None and self.obs_history_len > 1:
            # Preserve the reset bank's temporal context. _obs() appends the
            # jittered current frame, replacing the bank's newest frame.
            frames = spin_obs.reshape(self.obs_history_len, 6)
            self._obs_history.extend(frames[:-1].copy())
            prior_actions = [float(x) for x in frames[:, 5]]
            smooth_seed = ([prior_actions[0]] * self.action_smooth_window
                           + prior_actions)[-self.action_smooth_window:]
            self._action_smooth_buf = deque(
                smooth_seed, maxlen=self.action_smooth_window
            )
            self._lagged_action = prev_action0
        elif prev_action0 != 0.0:
            # Initialise smoothing history consistently with prev_action.
            self._action_smooth_buf = deque(
                [prev_action0] * self.action_smooth_window,
                maxlen=self.action_smooth_window,
            )
            self._lagged_action = prev_action0
        mujoco.mj_forward(self.model, self.data)
        if self.firmware_obs_model:
            self.seed_measurement_ring()
        return self._obs(), {}

    def seed_measurement_ring(self) -> None:
        """(Re)fill the measurement ring from the CURRENT joint state.

        Back-extrapolates positions with the current joint velocities so the
        first window finite-difference reads the true initial velocity
        (matters for near-upright resets, which start with spin — the real
        firmware's buffer holds genuine motion at engage). Also used by
        replay tooling after overwriting qpos/qvel directly.
        """
        motor0 = float(self.data.qpos[self._motor_qpos_addr])
        pen0 = float(self.data.qpos[self._pen_qpos_addr])
        motor_vel0 = float(self.data.qvel[self._motor_qvel_addr])
        pen_vel0 = float(self.data.qvel[self._pen_qvel_addr])
        self._prev_cmd_pos = motor0  # commanded path starts at the true position
        dt_p = self.model.opt.timestep
        n = self._meas_ring.maxlen
        self._meas_ring.clear()
        self._meas_ring.extend(
            (motor0 - motor_vel0 * (n - 1 - k) * dt_p,
             pen0 - pen_vel0 * (n - 1 - k) * dt_p)
            for k in range(n)
        )
        self._capture_measured_state()

    def set_actuator_state(
        self,
        position_rad: float,
        *,
        planner_velocity_rad_s: float = 0.0,
        host_velocity_cmd_rad_s: float = 0.0,
    ) -> None:
        """Synchronise MuJoCo helpers and the queued planner for replay tools."""
        self._motor_target = float(position_rad)
        self._lagged_target = float(position_rad)
        self._prev_cmd_pos = float(position_rad)
        self._motor_vel = float(planner_velocity_rad_s)
        self._host_velocity_cmd = float(host_velocity_cmd_rad_s)
        self._accel_transport.configure(
            self._transport_delay_s,
            self._planner_speed_limit_rad_s,
        )
        self._accel_transport.reset(
            position_rad=position_rad,
            velocity_rad_s=planner_velocity_rad_s,
        )

    def mirror_actuator_state(self) -> None:
        """Reflect command, planner, and transport-filter state in place."""
        self._motor_target = -self._motor_target
        self._lagged_target = -self._lagged_target
        self._prev_cmd_pos = -self._prev_cmd_pos
        self._motor_vel = -self._motor_vel
        self._host_velocity_cmd = -self._host_velocity_cmd
        self._lagged_action = -self._lagged_action
        self._lagged_accel_cmd = -self._lagged_accel_cmd
        self._action_queue = deque(-float(v) for v in self._action_queue)
        self._accel_cmd_queue = deque(-float(v) for v in self._accel_cmd_queue)
        self._accel_transport.mirror()

    def kick(self, value: float, ticks: int = 2) -> None:
        """Override the simulated actuator's normalized command for `ticks` steps."""
        self._kick_value = float(np.clip(value, -1.0, 1.0))
        self._kick_ticks_left = max(0, int(ticks))

    def step(self, action):
        action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        # Optional action smoothing precedes transport. Reward and observation
        # retain the raw action. The lab wrapper disables smoothing.
        if self.action_smooth_window > 1:
            self._action_smooth_buf.append(action)
            smoothed_action = sum(self._action_smooth_buf) / self.action_smooth_window
        else:
            smoothed_action = action

        # A kick overrides the actuator command after smoothing; the reported
        # previous action and reward still use the controller's own output.
        if self._kick_ticks_left > 0:
            smoothed_action = self._kick_value
            self._kick_ticks_left -= 1

        # --- Continuous action lag: first-order LP filter on the action. ---
        # Models the laptop ↔ Arduino ↔ stepper-ISR pipeline as a low-pass.
        # The rational discretisation `alpha = dt / (tau + dt)` makes the
        # filter behave like a "fractional-step delay": tau=0 ⇒ alpha=1 ⇒
        # no lag; tau=dt ⇒ alpha=0.5 ⇒ half current + half previous.
        if self._action_lag_tau_s > 0.0:
            dt_ctrl = 1.0 / self.control_freq_hz
            alpha = dt_ctrl / (self._action_lag_tau_s + dt_ctrl)
            self._lagged_action = (
                (1.0 - alpha) * self._lagged_action + alpha * smoothed_action
            )
            lagged_action = self._lagged_action
        else:
            self._lagged_action = smoothed_action
            lagged_action = smoothed_action

        # Optional integer-step delay follows the low-pass action filter.
        if self._action_delay_steps > 0:
            self._action_queue.append(lagged_action)
            delayed_action = float(self._action_queue.popleft())
        else:
            delayed_action = lagged_action

        n_sub = self._n_substeps
        actual_dt_s = n_sub * self.model.opt.timestep

        host_accel_cmd = 0.0
        planner_accel_cmd = 0.0
        active_accel_cmd = 0.0
        desired_velocity_cmd = 0.0
        if self.action_mode in ("accel", "velocity"):
            # Queued pulses continue until the delayed command reaches the
            # planner; its speed cap is separate from the host request limit.
            if self.action_mode == "velocity":
                v_des = smoothed_action * self.max_velocity_rad_s
                desired_velocity_cmd = v_des
                host_accel_cmd = float(np.clip(
                    (v_des - self._host_velocity_cmd) * self.control_freq_hz,
                    -self.max_accel_rad_s2,
                    self.max_accel_rad_s2,
                ))
                self._host_velocity_cmd = float(np.clip(
                    self._host_velocity_cmd
                    + host_accel_cmd / self.control_freq_hz,
                    -self.max_velocity_rad_s,
                    self.max_velocity_rad_s,
                ))
                if self.firmware_obs_model:
                    self._host_velocity_cmd += 0.1 * (
                        self._meas_motor_vel - self._host_velocity_cmd
                    )

                # These optional filters must be disabled when using explicit
                # transport delay, as checked during initialization.
                transported_accel = host_accel_cmd
                if self._action_lag_tau_s > 0.0:
                    dt_ctrl = 1.0 / self.control_freq_hz
                    alpha = dt_ctrl / (self._action_lag_tau_s + dt_ctrl)
                    self._lagged_accel_cmd = (
                        (1.0 - alpha) * self._lagged_accel_cmd
                        + alpha * transported_accel
                    )
                    transported_accel = self._lagged_accel_cmd
                if self._action_delay_steps > 0:
                    self._accel_cmd_queue.append(transported_accel)
                    transported_accel = float(self._accel_cmd_queue.popleft())
            else:
                host_accel_cmd = delayed_action * self.max_accel_rad_s2
                transported_accel = host_accel_cmd

            planner_accel_cmd = float(np.clip(
                transported_accel,
                -self._motor_max_accel_rad_s2,
                self._motor_max_accel_rad_s2,
            ))
            if (self._motor_target >= MOTOR_SAFE_LIMIT_RAD
                    and planner_accel_cmd > 0.0):
                planner_accel_cmd = 0.0
            elif (self._motor_target <= -MOTOR_SAFE_LIMIT_RAD
                  and planner_accel_cmd < 0.0):
                planner_accel_cmd = 0.0
            self._accel_transport.command(planner_accel_cmd)
        else:
            # --- Position-delta integration: action → per-step target delta. ---
            # motor_target += action × MAX_ACTION_DELTA_RAD,
            # then moveTo(target). `_motor_target` accumulates the deltas (clamped
            # to the rail); the stepper's ramp bandwidth is a first-order lag
            # (`_motor_tau_s`), so the PD actuator chases `_lagged_target`. tau=0 ⇒
            # no lag.
            self._motor_target = float(np.clip(
                self._motor_target + delayed_action * self.max_action_delta_rad,
                -MOTOR_SAFE_LIMIT_RAD,
                MOTOR_SAFE_LIMIT_RAD,
            ))
            if self._motor_tau_s > 0.0:
                alpha = actual_dt_s / (self._motor_tau_s + actual_dt_s)
                self._lagged_target = (
                    (1.0 - alpha) * self._lagged_target + alpha * self._motor_target
                )
            else:
                self._lagged_target = self._motor_target
            cmd_end = self._lagged_target

        cmd_start = self._prev_cmd_pos
        physics_dt = self.model.opt.timestep
        for k in range(n_sub):
            if self.action_mode in ("accel", "velocity"):
                cmd_k, active_accel_cmd = self._accel_transport.advance(
                    physics_dt
                )
                self._accel_transport.clamp_position(
                    -MOTOR_SAFE_LIMIT_RAD, MOTOR_SAFE_LIMIT_RAD
                )
                cmd_k = self._accel_transport.position_rad
            else:
                # Position mode retains continuous target interpolation.
                cmd_k = cmd_start + (cmd_end - cmd_start) * (k + 1) / n_sub
                active_accel_cmd = 0.0
            self.data.ctrl[0] = cmd_k
            mujoco.mj_step(self.model, self.data)
            if self.firmware_obs_model:
                self._meas_ring.append((
                    cmd_k,
                    float(self.data.qpos[self._pen_qpos_addr]),
                ))

        if self.action_mode in ("accel", "velocity"):
            self._motor_target = self._accel_transport.position_rad
            self._motor_vel = self._accel_transport.velocity_rad_s
            cmd_end = self._motor_target
        if self.firmware_obs_model:
            self._capture_measured_state()

        self._prev_cmd_pos = cmd_end
        self._step_count += 1

        motor_pos_now = float(self.data.qpos[self._motor_qpos_addr])
        terminated = False
        hard_stop = False
        reward = self._reward(action)
        self._prev_action = float(action)
        self._prev_motor_vel = float(self.data.qvel[self._motor_qvel_addr])
        if self.terminate_on_hard_stop and abs(motor_pos_now) >= MOTOR_LIMIT_RAD:
            terminated = True
            hard_stop = True
            reward -= self.hard_stop_penalty

        # Count accumulated travel across angle wraps. Holding the upright
        # gate for one second clears the budget; a pass-through does not.
        phi_now = float(self.data.qpos[self._pen_qpos_addr])
        self._spin_travel_rad += abs(_wrap_pi(phi_now - self._spin_last_phi))
        self._spin_last_phi = phi_now
        theta_now = _wrap_pi(phi_now - math.pi)
        pen_vel_now = float(self.data.qvel[self._pen_qvel_addr])
        if abs(theta_now) <= math.radians(15.0) and abs(pen_vel_now) <= 4.0:
            self._spin_catch_ticks += 1
        else:
            self._spin_catch_ticks = 0
        if self._spin_catch_ticks >= int(math.ceil(self.control_freq_hz)):
            self._spin_caught = True
            self._spin_travel_rad = 0.0
        spin_loop = (
            self._spin_travel_rad
            >= self.spin_loop_revolutions * 2.0 * math.pi
        )
        spin_revolutions = self._spin_travel_rad / (2.0 * math.pi)
        spin_excess = max(
            spin_revolutions - self.spin_progress_free_revolutions, 0.0
        )
        spin_progress_penalty = (
            self.spin_progress_penalty_weight * spin_excess * spin_excess
        )
        reward -= spin_progress_penalty
        if self.terminate_on_spin_loop and spin_loop and not terminated:
            terminated = True
            reward -= self.spin_loop_penalty

        truncated = self._step_count >= self._max_steps
        info = {
            "motor_pos": motor_pos_now,
            "phi": float(self.data.qpos[self._pen_qpos_addr]),
            "accel_cmd_rad_s2": host_accel_cmd,
            "desired_velocity_rad_s": desired_velocity_cmd,
            "planner_accel_cmd_rad_s2": planner_accel_cmd,
            "active_accel_rad_s2": active_accel_cmd,
            "motor_target": self._motor_target,
            "motor_vel_cmd": self._motor_vel,
            "host_velocity_cmd_rad_s": self._host_velocity_cmd,
            "planner_velocity_rad_s": self._motor_vel,
            "transport_delay_s": self._transport_delay_s,
            "motor_planner_speed_limit_rad_s": self._planner_speed_limit_rad_s,
            "motor_max_accel_rad_s2": self._motor_max_accel_rad_s2,
            "action_delay_steps": self._action_delay_steps,
            "action_lag_tau_s": self._action_lag_tau_s,
            "smoothed_action": smoothed_action,
            "motor_tau_s": self._motor_tau_s,
            "obs_staleness_s": self._obs_staleness_s if self.firmware_obs_model else 0.0,
            "motor_vel_meas": self._meas_motor_vel if self.firmware_obs_model else None,
            "spin_loop": bool(spin_loop),
            "spin_revolutions_since_catch": spin_revolutions,
            "spin_progress_penalty": spin_progress_penalty,
            "sustained_upright_catch": self._spin_caught,
            "hard_stop": hard_stop,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            from mujoco import viewer  # noqa: WPS433  (deferred import)
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # --- Internal helpers --------------------------------------------------

    def _theta_upright(self) -> float:
        """Pendulum angle measured from upright. theta=0 -> upright."""
        # MJCF: joint angle phi=0 is hanging-down. Upright is phi=pi.
        # theta = phi - pi  (so theta=0 at upright, theta=±pi at hanging down).
        phi = float(self.data.qpos[self._pen_qpos_addr])
        return _wrap_pi(phi - math.pi)

    def _capture_measured_state(self) -> None:
        """Model one GET_STATE read from the ring of physics-rate positions.

        Positions are quantised to the step-counter / AS5600 resolution and
        stale by `_obs_staleness_s`; velocities are finite differences of the
        quantised positions across the firmware's 8 ms window, so the
        LSB-scale spikes emerge from the same mechanism as on the rig.
        """
        dt_p = self.model.opt.timestep
        ring = self._meas_ring
        k_stale = int(round(self._obs_staleness_s / dt_p))
        idx_new = max(0, len(ring) - 1 - k_stale)
        idx_old = max(0, idx_new - self._meas_win_substeps)
        m_new, p_new = ring[idx_new]
        m_old, p_old = ring[idx_old]
        mq_new = round(m_new / MOTOR_STEP_LSB_RAD) * MOTOR_STEP_LSB_RAD
        mq_old = round(m_old / MOTOR_STEP_LSB_RAD) * MOTOR_STEP_LSB_RAD
        pq_new = round(p_new / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        pq_old = round(p_old / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        span_s = max(1, idx_new - idx_old) * dt_p
        self._meas_motor_pos = mq_new
        self._meas_pen_phi = pq_new
        self._meas_motor_vel = (mq_new - mq_old) / span_s
        self._meas_pen_vel = (pq_new - pq_old) / span_s

    def _obs(self) -> np.ndarray:
        if self.firmware_obs_model:
            # Measured-state path: what the host actually reads back.
            motor_pos = self._meas_motor_pos
            phi = self._meas_pen_phi
            motor_vel = self._meas_motor_vel
            pen_vel = self._meas_pen_vel
        else:
            motor_pos = float(self.data.qpos[self._motor_qpos_addr])
            phi = float(self.data.qpos[self._pen_qpos_addr])
            motor_vel = float(self.data.qvel[self._motor_qvel_addr])
            pen_vel = float(self.data.qvel[self._pen_qvel_addr])

        theta = _wrap_pi(phi - math.pi)
        if self.obs_include_velocities:
            frame = np.array(
                [motor_pos, math.sin(theta), math.cos(theta), motor_vel,
                 pen_vel, self._prev_action],
                dtype=np.float32,
            )
        else:
            frame = np.array(
                [motor_pos, math.sin(theta), math.cos(theta),
                 self._prev_action],
                dtype=np.float32,
            )
        # Frame stacking: called exactly once per reset/step, so appending
        # here keeps the history in lockstep with the control ticks.
        if not self._obs_history:
            self._obs_history.extend([frame] * self.obs_history_len)
        else:
            self._obs_history.append(frame)
        if self.obs_history_len == 1:
            return frame
        return np.concatenate(self._obs_history)

    def _reward(self, action: float) -> float:
        # Evaluate the shared reward using the current state and action history.
        return compute_reward(
            theta=self._theta_upright(),
            pen_vel=float(self.data.qvel[self._pen_qvel_addr]),
            motor_pos=float(self.data.qpos[self._motor_qpos_addr]),
            motor_vel=float(self.data.qvel[self._motor_qvel_addr]),
            action=action,
            prev_action=self._prev_action,
            prev_motor_vel=self._prev_motor_vel,
            weights=self._reward_weights,
        )


def _wrap_pi(x: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi
