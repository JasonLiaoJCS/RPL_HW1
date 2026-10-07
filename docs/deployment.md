# Firmware and Deployment

Use the firmware supplied with this release. No controller runs on
the Nano: it provides encoder feedback and step generation; student code runs
on the host at 50 Hz.

## Flash and inspect with 12 V off

Connect USB with the arm physically centered and pendulum hanging still. Opening
the serial port can reset the Nano and establish a new angle origin. Reported
zero alone does not establish physical centering.

```bash
pendulum-lab ports
# Replace /dev/ttyUSB0 with the device listed above (or a stable /dev/serial/by-id path).
pio run -d firmware/TestHeartbeat -t upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

Exit the monitor with Ctrl-C before uploading another sketch. The heartbeat LED
double-blinks; it is only a USB/firmware check.

```bash
pio run -d firmware/TestEncoder -t upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

Verify magnet detection and response as you turn the pendulum by hand. Restore
the hanging pose before loading the binary server:

```bash
pio run -d firmware/LowLevelServer -t upload --upload-port /dev/ttyUSB0
pendulum-lab preflight --port /dev/ttyUSB0
```

The supplied PlatformIO environment uses `nanoatmega328`. Some kit Nanos have a
different bootloader; have the TA select the matching PlatformIO board instead
of repeatedly changing controller code. Close all serial monitors before using
`pendulum-lab`. If uploads fail, check the device path, permissions, bootloader,
cable and port ownership. On Linux, membership in the device's serial-access
group is normally required; log out/in after an administrator changes it.

The release server advertises protocol **3**, 32 microsteps and 500000 baud. It
rejects acceleration commands beyond +/-50 rad/s^2. In acceleration mode, 100 ms
without a valid command aborts queued motion and disables outputs. READY and
feedback reads do not keep the motor alive. Finite position/calibration moves
use bounded targets rather than that command watchdog. The Python preflight
rejects older firmware without this release contract.

Protocol 3 checks the +/-100-degree step-count boundary in the MCU loop,
independently of host acceleration commands. Encoder read errors, missing magnet,
implausible angle increments and I2C timeouts latch a motor-disabled fault.
Faults prevent re-engagement until the board is reset; do not reset and retry
without inspecting the cause. Fault bits are travel=1, encoder=2, watchdog=4,
invalid command=8. Failed encoder samples are not reported as fresh measurements.
These are software protections, not physical limit switches. Missed steps,
incorrect scaling and unpowered coasting can still invalidate physical travel.

## Verify the motor

For USB-only free-swing identification, follow the
[system-identification guide](system-identification.md). Keep 12 V off and
mechanically secure the arm; a fitted pendulum model does not replace this motor check.

Power OFF, center the arm and let the pendulum hang still, then run:

```bash
pendulum-lab motor-check --port /dev/ttyUSB0
```

After preflight, the terminal asks you to power ON and type `MOVE`. It requests
one finite +533-microstep move, approximately +30 degrees, then disables the
motor. Inspect motion while it runs; turn power off before touching or returning
the arm to center. This verifies physical movement only when your observation
agrees with the step count. It is not an encoder-based arm measurement.

## Deploy a student controller

First verify the student's implementation and action scaling in simulation.
Verify hanging-start operation in simulation, not only near-upright balancing.
Do not hold the mechanism upright while a powered controller runs. The shared
hardware runner starts only from a centered, still, hanging pose.

```bash
# Power OFF, center physically, let pendulum hang, then start the command.
pendulum-lab hardware --algorithm lqr --port /dev/ttyUSB0 --duration 3 --output runs/lqr_hw_001

# After successful commissioning, repeat from the same prepared starting state.
pendulum-lab hardware --algorithm lqr --port /dev/ttyUSB0 --duration 15 --output runs/lqr_hw_002
```

Add `--config your_settings.json`, `--params your_rig.json` and, for a trained RL
controller, `--checkpoint your_policy.pt`. `--duration` accepts up to 30 seconds,
the length of a demo run. At the prompt, power ON, clear the
motion area, keep the switch accessible and type `RUN`. Preflight and the run use
the same serial connection. Each invocation performs exactly one hardware trial.

The runner checks the supplied firmware profile, fresh timestamps, finite state,
reported arm angle strictly between -100 and +100 degrees, arm speed under 6 rad/s
(the firmware itself caps speed near 5 rad/s), and feedback/control
lateness under 40 ms. Exceptions and Ctrl-C request disengagement. On completion,
switch 12 V OFF before recentering. With motor power off, USB can stay connected.

The watchdog disables torque; it does not catch a falling pendulum. Host cleanup
is best effort, and the open-loop step counter cannot reveal missed steps. Clear
the full pendulum swing area and use the physical switch when needed. No online
learning or automatic repeated powered trials are supplied.

Inspect `summary.json` for the stop reason: `duration` is a planned shutdown;
guards, timing errors and controller exceptions are failed attempts. Success
still requires the full balance metric, not merely reaching the time limit.
