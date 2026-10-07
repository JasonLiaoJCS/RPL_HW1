# System Identification

## Responsibilities

- The complete acquisition and fitting implementation is supplied.
- You assemble the rig, perform the experiments, inspect the results and justify model quality.
- Collect your own measurements. No instructor recordings or calibrated parameter files are included.
- Submit raw recordings, experimental notes, fitted JSON, comparison plots and limitations.
- Record a video of the identification process for `Report.md`.

## Supplied Workflow

1. Record a three-second hanging-zero tare.
2. Record three ten-second free-swing trials at 100 Hz, with the arm fixed and motor disabled.
3. Optionally inspect a powered `0, +90, 0, -90, 0` degree motor sweep.
4. Fit oscillation periods and the combined viscous/dry-friction peak envelope.
5. Aggregate the trials and derive friction using the packaged CAD geometry.
6. Export a per-rig JSON and compare a recorded free swing with the simulator.
7. Capture acceleration step/chirp responses and replay the applied commands.
8. Record feedback-driven motor-envelope trials and fit driven coupling.

`sysid_core.py` contains the fitting formulas. `sysid_wizard.py` provides the
interactive experiment, file handling and plots. The fit is a peak-envelope
fit, not a nonlinear trajectory optimization.

## Prepare

- Install the dependencies using the [README](../README.md); Matplotlib is included for plots.
- Complete [assembly](hardware.md) and [firmware commissioning](deployment.md).
- Switch **12 V OFF**; leave USB connected, arm centered and pendulum hanging motionless.
- Keep 12 V off for tare and free-swing recording. Software cannot verify the supply is off.
- No powered experiment is retried automatically after a failure.
- Check that the magnet is seated and that the pendulum moves freely.
- Mass, COM and COM inertia come from `src/pendulum_lab/plant/data/model.urdf`.
- Check this geometry against your actual assembly. A changed tip mass or print may need an updated model.
- The fitting command does not estimate mass and COM from free swings or request them as numeric inputs.
- The kit pendulum carries an NT$10 coin (7.5 g); the URDF assumes a 7.12 g coin. After fitting,
  weigh the complete pendulum and replace `run_meta.pendulum_total_mass_kg` in your JSON with the measured value.

## Collect And Fit

```bash
pendulum-lab ports

# Full interactive workflow, including the separately confirmed motor sweep.
pendulum-lab sysid --port /dev/ttyUSB0 \
  --out-dir runs/sysid_train_01 --out-json configs/my_rig.json

# Free-swing-only workflow: keep motor power off throughout.
pendulum-lab sysid --port /dev/ttyUSB0 --skip-motor \
  --out-dir runs/sysid_train_02 --out-json configs/my_rig_02.json
```

- Replace the port with your detected device. If omitted, the wizard lists candidate USB ports.
- Follow the hanging-zero prompt before moving the pendulum.
- For each free-swing trial, hold or mechanically secure the arm without obstructing the pendulum.
- Support the displaced pendulum on a box. After the countdown, slide the box out perpendicular to its swing plane.
- Do not push the pendulum, obstruct its motion or allow the arm to move.
- Each output directory and parameter filename must be new; failed recordings are not silently retried.
- Options: `--n-free-runs 3`, `--duration-s 10`, `--sample-hz 100`.
- The lab policy rate remains 50 Hz. The faster recording rate does not change that contract.

## Separate Acquisition And Fitting

```bash
# USB needed. No powered motor sweep in this example.
pendulum-lab sysid collect --port /dev/ttyUSB0 --skip-motor \
  --out-dir runs/sysid_train_03

# Offline: no USB or hardware required.
pendulum-lab sysid fit --in-dir runs/sysid_train_03 \
  --out-json configs/my_rig_03.json
```

The recording directory contains:

- `metadata.json`: tare offset, settings and completion status.
- `tare.npz`: hanging-zero recording.
- `free_run_1.npz`, etc.: arrays `t`, `pen`, `motor`, `pvel`, `mvel`.
- `motor_sweep.npz`, when requested: time, step-count angle/velocity, acceleration and target.
- `freeswing_compare.png`, `.csv` and `.json` after fitting.
- `motor_sweep.png` when a sweep was recorded.

Times are host-monotonic seconds. Angles are radians; pendulum angles become
hanging-zero after subtracting `hanging_zero_rad`. MCU timestamps are checked
during acquisition for stale/discontinuous feedback. Arm feedback is an
open-loop step count, not a physical arm encoder.

## Fit And Interpret

- The fitter uses same-side extrema for period and an elliptic finite-angle period correction.
- It fits a viscous-plus-dry-friction peak-envelope recurrence and aggregates runs by medians.
- Friction derivation uses **CAD-predicted pivot inertia**, as in the supplied derivation.
- Period-derived inertia is exported alongside CAD inertia for comparison; the simulator loader uses the measured-inertia field.
- Inspect disagreement between the two inertias, warning messages and simulated/measured decay.
- Repeating the experiment and explaining discrepancies is part of your work.
- A same-recording comparison plot is a fit diagnostic, not an independent validation result.
- The free-swing fit alone does not identify motor response or driven coupling. Use the experiments below to investigate those separately.

## Independent Validation

Collect another free-swing session with different releases, then compare it with
the already fitted model without refitting:

```bash
pendulum-lab sysid collect --port /dev/ttyUSB0 --skip-motor \
  --out-dir runs/sysid_holdout_01
pendulum-lab sysid validate --in-dir runs/sysid_holdout_01 \
  --params configs/my_rig.json --out-dir runs/sysid_validation_01
```

- The tool rejects recorded files already used for fitting.
- Each held-out trial gets a simulator comparison plot, CSV and angular RMSE.
- Simulation is initialized at detected release using recorded position and velocity.
- Physical parameters remain frozen. Plots explicitly load the selected JSON, not the starter parameters.
- Inspect phase drift, decay mismatch, rest-angle calibration and repeatability.
- Explain whether the model is adequate for controller design and what it does not capture.

## Motor Check

```bash
pendulum-lab sysid validate-motor --port /dev/ttyUSB0 \
  --out-dir runs/sysid_motor_01
```

- Begin with 12 V off. Remove supports, center the arm and let the pendulum hang still.
- Preflight runs before the wizard asks you to power on and type `SWEEP`.
- Clear the entire motion area and keep the power switch accessible.
- PD acceleration control uses fixed gains and targets, capped at 40 rad/s^2.
- Watchdog commands continue during the holds. Firmware faults, timing misses, arm limits and settling timeouts abort the sweep.
- Observe both physical 90-degree endpoints. Software step counts cannot prove movement or detect missed steps.
- Power off before touching or recentering. Do not deploy if physical travel is unverified.

## Acceleration Response

```bash
# Offline first; no USB connection is opened.
pendulum-lab sysid accel --waveform step --skip-real \
  --params configs/my_rig.json --out-dir runs/accel_step_sim

# Attended capture. Start with 12 V OFF and follow the power prompts.
pendulum-lab sysid accel --port /dev/ttyUSB0 --waveform step \
  --params configs/my_rig.json --out-dir runs/accel_step_real

# Inspect the second waveform offline before collecting it on hardware.
pendulum-lab sysid accel --waveform chirp --skip-real \
  --params configs/my_rig.json --out-dir runs/accel_chirp_sim

# Replay an existing capture with a selected model; no hardware is opened.
pendulum-lab sysid accel --replay runs/accel_step_real/capture.npz \
  --params configs/my_rig.json --out-dir runs/accel_step_replay
```

- `sysid_accel.py` supplies the step sequence and chirp phase law.
- Step duration: 3.5 seconds. Chirp duration: 9 seconds.
- Step default scale: `1/3`, giving pulse magnitudes 16.67, 33.33 and 50 rad/s^2.
- Chirp default scale: `0.1`, giving a 10 rad/s^2 peak. `--scale` changes the explicit multiplier.
- Commands above 50 rad/s^2 are rejected, not silently clipped.
- Commands refresh at 50 Hz; feedback defaults to 200 Hz (`--sample-hz`, range 50-200).
- Short pulses are consequently sampled on the command clock, not applied continuously.
- The chirp phase is `2*pi*(0.5 + 2.5*s/8)*s`, for `0 <= s <= 8`.
- Its instantaneous frequency sweeps 0.5-5.5 Hz.
- The pendulum must swing freely. Never hold the arm or pendulum during a powered experiment.
- An open-loop waveform is not guaranteed to stay centered. These probes use a conservative 60-degree abort boundary inside the policy's 100-degree limit.
- Inspect physical motion; the motor counter cannot detect missed steps.

Outputs:

- `capture.npz`: seven-column feedback trace, MCU timestamps, actual command times, commands, initial state, scale and completion flag.
- `comparison.npz`: `t`, `accel_cmd`, `real_motor`, `real_pen`, `sim_motor`, `sim_pen`, `waveform`.
- `comparison.png`: command, arm and pendulum overlays.
- `metadata.json`: settings, completion status and angular errors when real data exist.
- Sim-only archives have empty real arrays, not fabricated measurements.
- Aborted captures remain marked incomplete and are rejected by offline replay.

Replay uses the selected parameter JSON, recorded initial position/velocity and
the actual command schedule. Commands are quantized to 5 ms simulator ticks;
results are interpolated to feedback receipt times. Both sides use the same
hanging-zero coordinate frame without sign inversion. The model retains the
lab's fixed transport delay and planner-speed settings; the replay does not
estimate them automatically.

## Motor Envelope And Coupling

```bash
# Begin with a single low-speed level; power OFF before launching.
pendulum-lab sysid motor-envelope --port /dev/ttyUSB0 \
  --speeds 1.5 --accels 25 --cycles 2 --out-dir runs/envelope_01

# Offline regression using your fitted free-swing JSON.
pendulum-lab sysid fit-coupling runs/envelope_01 \
  --params configs/my_rig.json --output runs/coupling_01.json
```

- `motor_envelope_probe.py` accelerates to a feedback velocity, reverses until near rest, then returns to center. Successive cycles alternate direction.
- Each level requires a fresh power-off preflight and explicit power-on confirmation.
- Physical return to the center mark, actual movement and absence of skipping must be confirmed. A failed check stops the ladder.
- Optional comma-separated `--speeds` and `--accels` select the ladder. Defaults: `1.5,2.5,3.5` rad/s and `25,50` rad/s^2.
- Two cycles are the default; up to ten can be requested. Each level has a hard 15-second duration limit, so large cycle counts may abort.
- Excessive predicted excursion is rejected before opening USB. Feedback travel is limited to 60 degrees; all normal firmware/state guards remain active.
- Center return uses bounded acceleration control with continuous watchdog refresh.
- Output files use a seven-column `samples` layout: host time, phase, acceleration command, arm angle, hanging-zero pendulum angle, arm velocity, pendulum velocity.
- Additional fields record command times, MCU times and completion. Negative phase IDs mark center return and are excluded from coupling fitting.
- `summary.json` keeps physical acceptance and paired speed/acceleration recommendations. These are not proof against missed steps.

The fitter uses these driven-regression equations:

```text
y = pendulum_acceleration + omega_n^2 * sin(phi) + damping * pendulum_velocity
X = [cos(phi) * arm_acceleration, sin(phi) * arm_velocity^2, 1]
```

- Velocity derivatives use Savitzky-Golay filtering separately within each phase; filter edges are discarded.
- Defaults test windows of 5, 7, 9, 11 and 15 samples; the report selects the window closest to 7.
- Reports include global and direction-specific coupling, constant bias, R-squared, sample counts and window sensitivity.
- Gravity frequency, inertia and viscous damping come from the free-swing JSON. Starter JSON without a fitted frequency is not sufficient.
- Mass and COM use explicit JSON metadata when present, otherwise CAD geometry; `pendulum_tip_mass` coin metadata is also read.
- `--arm-length-m` defaults to the 65 mm arm (0.065 m).
- Seven-column NPZ files without completion metadata remain readable, but cannot prove that the experiment passed its checks.
- Explicitly failed/incomplete sessions, invalid timestamps and rank-deficient experiments are rejected.
- The coupling report is a diagnostic; it does not silently rewrite the plant JSON or apply a gain to the simulator.
- This is not a full torque, backlash, missed-step or transport-delay estimator. Inspect residuals and explain model limitations.

## Use Your Model

Pass `--params configs/my_rig.json` to subsequent simulation, training and
deployment commands. A good free-swing fit does not replace the hardware
commissioning checklist or prove that a policy will balance the physical rig.
