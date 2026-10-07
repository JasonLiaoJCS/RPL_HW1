# Assignment

Build the pendulum and implement PID, LQR, iLQR, MPC, PPO, SAC and DQN. Use the
supplied infrastructure so all methods share the same physical task. The starter
is deliberately coarse: you are responsible for deriving the algorithms,
implementing them, selecting parameters, testing them and explaining the results.
Library linear algebra, differentiation and optimization are allowed. PPO, SAC
and DQN may build on any deep-RL library (Stable-Baselines3 is recommended).
Derive the classical controllers' A and B matrices yourself. Ask the TA before
adding other packages, and document their installation in the README.

## System identification

Identify a model of your own assembled pendulum before using it for controller
design or training. Use the supplied [system-identification tools](system-identification.md)
to record free swings, fit pendulum parameters, inspect driven step/chirp responses
and estimate coupling from motor-envelope trials. Perform
the physical experiments yourself and justify your measurements and experimental
conditions. Validate your model against measurements not used to fit it,
and explain its limitations. Follow the hardware commissioning and attended-test
requirements when collecting data.

Submit your experiment commands, any code modifications, recorded data, estimated parameters and
validation results, and record a video of the identification process for `Report.md`.
Use your parameter file via `--params` in subsequent runs.
The supplied `example_rig.json` contains deliberately uncalibrated placeholder
values, not measurements of your kit. Replace them with your own estimates;
matching this starter simulator alone does not validate a physical model.
The fitting implementation is supplied; collecting valid experimental evidence
and explaining the model's limitations remain your responsibility. The supplied
free-swing fit does not identify the motor or transport-delay model. Separate
driven-response acquisition and coupling fitting are supplied; inspect their
limitations rather than treating any fitted report as proof of hardware readiness.

## Controller interface

Each algorithm file exports:

```python
class Controller:
    def __init__(self, *, params_path, config, checkpoint=None): ...
    def reset(self): ...
    def act(self, observation): ...
```

`config` is your JSON dictionary. `params_path` identifies the rig's physics.
`checkpoint` is an optional path; your implementation owns its format and loading.
Initialization should do expensive model construction and optimization setup.
`reset()` clears episode state. `act()` must return a scalar normalized
acceleration in [-1, 1]. The runner clips finite scalar outputs and rejects NaN,
infinity or arrays with multiple entries. DQN must map its discrete action index
to this same scalar interface.

`observation` has shape `(24,)`: four oldest-to-newest frames, each
`[arm, sin(theta), cos(theta), arm_velocity, pendulum_velocity, previous_action]`.
Angles use radians, velocities rad/s, and `theta = wrap(phi - pi)` is zero
upright. The encoder's `phi` is zero hanging. The initial history repeats the
first frame, with previous action zero. `contracts.state_from_obs()` returns
`[arm, theta, arm_velocity, pendulum_velocity]` from the newest frame.

An action of +1 means +50 rad/s^2 commanded arm acceleration, **not torque,
velocity or position**. One tick is 0.02 s. Zero acceleration does not stop a
moving motor. The full simulator includes queued actuator delay and firmware-like
observations; a four-state ideal model is only an approximation. Inspect the
supplied plant when constructing predictive controllers.

The lecture slides write the arm angle as φ and the pendulum angle as θ; here
they are `arm` and `theta`. The raw hanging-zero encoder angle that the firmware
and sysid tools call `phi` is not the lecture's φ. The lecture model takes motor
torque `u`, but this lab commands arm acceleration: φ̈ itself is the input
(50 rad/s^2 per unit action), so the arm inertia J_r drops out. The lecture
equations also measure θ in the opposite direction to this simulator. Substitute
θ → −θ (equivalently, flip the sign of C) before deriving A and B.

- Arm travel: +/-100 degrees; reaching either boundary stops the run.
- Arm speed: the stepper planner caps it at 5 rad/s, in simulation and firmware.
  A guard stops the run above 6 rad/s, which only a fault should cause.
- Start with the arm physically centered; check clearance and wire slack with power off.

## What to implement

| File | Expected work |
| --- | --- |
| `algorithms/pid.py` | Simple PID stabilization; explain gains, saturation and integral handling. |
| `algorithms/lqr.py` | Model/linearization, Riccati solution (continuous as in the lecture, or discrete at 50 Hz), choice of Q/R and stability checks. |
| `algorithms/ilqr.py` | Nonlinear trajectory optimization and application of the planned controls. |
| `algorithms/mpc.py` | Receding-horizon optimization with actuator/arm constraints and timing measurements. |
| `algorithms/ppo.py` | On-policy learning, checkpointing and deterministic evaluation. |
| `algorithms/sac.py` | Continuous off-policy learning, checkpointing and deterministic evaluation. |
| `algorithms/dqn.py` | Discrete action design, value learning, checkpointing and deterministic evaluation. |
| `algorithms/custom.py` | Optional: your own algorithm, hybrid controller or other experiment. |

- The seven named algorithms remain required; `custom.py` is an extra workspace.
- Choose any approach for `custom.py`; keep its implementation and helpers in that file.
- Keep each required algorithm's implementation and helpers in its provided file too.
- Keep the same controller interface, evaluation metrics and hardware safeguards.
- Select it with `--algorithm custom` in `sim`, `hardware` or `train`.
- Implement its `train(...)` hook only if your method needs training.
- Paper references are listed in the [README](../README.md#references).

PID and upright LQR are local stabilizers. Evaluate local balancing with
`--start upright`, but remember that hardware starts with the pendulum hanging
down. Does your model remain valid throughout that motion? How will your
controller reach the region where it can balance? Develop and justify your own
approach. Report near-upright and hanging-start results separately, and identify
any additional control logic rather than presenting it as pure LQR.

Swing-up logic is allowed for the classical controllers (PID, LQR, iLQR, MPC)
only. Any method in code is acceptable, even a hard-coded one, but the graded
algorithm must do the balancing, and `Report.md` must describe the swing-up.
PPO, SAC and DQN policies swing up and balance on their own.

RL training uses `train(*, env_factory, config, output_dir)`. Calling
`env_factory()` returns a fresh continuous-action Gymnasium environment. Write
the training loop yourself or build it on a deep-RL library such as
Stable-Baselines3; either way, export checkpoints and learning curves. Respect
the difference between `terminated` and `truncated`. Seed your own RNGs and the
environment; the runner does not seed your student code for you. Store the lab
contract, plant identity, action mapping and dependencies with the checkpoint and
validate them when loading. The infrastructure does not assume a checkpoint format.
`.gitignore` excludes `runs/`, `*.pt`, `*.zip` and `*.pkl`, so commit the final
checkpoint you submit explicitly (see the README's submission section).

## Experiments and submission

1. Document your assembled rig: part variations, tip mass, driver settings,
   wiring, physical +30-degree check, encoder readings and free-swing behavior.
2. Test near-upright stabilization and hanging-start operation separately. Use fixed seeds
   for development and different seeds for final evaluation. Include failures.
3. Compare all seven methods in simulation. Record balance, arm excursion,
   command magnitude, computation time, training effort and sample efficiency.
4. Deploy every algorithm in attended trials once it works in simulation. Begin
   with 3 seconds, then use 15 seconds after a successful commissioning run.
   Hardware runs accept up to 30 seconds, the length of a demo run.
5. Submit source, identification evidence (including a video of the process),
   configuration, environment versions, checkpoints, simulator GIFs and results,
   real traces/video and `Report.md` explaining sim-to-real differences. The
   README lists what `Report.md` must cover.

The common full-task metric is a complete episode of at least 15 seconds whose
final 10 seconds remain within 15 degrees of upright and 4 rad/s pendulum speed,
with no guard failure. A short trial cannot pass this metric. The simulator evaluates
true physical state; hardware metrics use encoder and step-count feedback.
Report local-stabilization metrics separately when evaluating upright starts.

Run artifacts contain `run.json`, `summary.json` and `episode_NNN.npz`. The NPZ
contains `samples` and named `columns`: time, arm angle, upright error, arm speed,
pendulum speed and normalized action. Count every attempted trial, including
early stops. Training completion or high reward alone does not establish a usable
controller.

The default simulator resets at a centered hanging pose. Upright starts randomize
pendulum error within about 17 degrees and arm position within 5 degrees. Mixed
starts split between those two distributions. Identify the conditions used in
your final evaluation.

## Hint: beyond one simulator

A controller that works on one simulated rig may not work on your assembled
pendulum. Is success in your current simulation enough? Consider what you might
change before deployment and how you would tell whether it helps.

You may extend the simulation and training code to investigate your ideas.
Keep the controller interface, evaluation criteria and hardware safeguards unchanged.
