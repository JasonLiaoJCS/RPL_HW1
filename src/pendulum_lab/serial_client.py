"""Python client for the LowLevelServer Arduino sketch.

Speaks the binary protocol defined at the top of
firmware/LowLevelServer/LowLevelServer.ino:
    0x01 READY               -> board echoes 0x01
    0x02 GET_STATE           -> board returns 20 bytes:
                                  uint32 time_us
                                  float32 motor_pos_rad
                                  float32 pendulum_pos_rad
                                  float32 motor_vel_rad_s
                                  float32 pendulum_vel_rad_s
    0x03 SET_ACCEL + float   commanded angular acceleration in rad/s²
    0x04 ENGAGE_MOTOR
    0x05 DISENGAGE_MOTOR
    0x07 SET_TARGET + float  commanded motor position in radians (position mode)
    0x08 ZERO_MOTOR          -> board echoes 0x08
    0x09 GET_PROFILE         -> command, version, microsteps, baud (7 bytes)
    0x0A CALIBRATE_30        -> finite +533 microsteps, command/status ack
    0x0B CALIBRATE_30_REVERSE -> finite -533 microsteps, command/status ack
    0x0C GET_MOTOR_STATUS    -> read-only MCU pin modes/levels and planner state
    0x0D GET_SAFETY          -> command, version=1, latched safety-fault bitmask

Sign convention: LowLevelServer sends motor and pendulum
positions and velocities without sign flips. This client passes them through;
deployment consumes that same frame. Motor position and velocity
are derived from step counts, not a physical arm encoder.

This client is request-driven (one 20-byte read per get_state call).
The lab firmware uses 500 kbaud.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

import serial


CMD_READY = 0x01
CMD_GET_STATE = 0x02
CMD_SET_ACCEL = 0x03  # commanded angular acceleration in rad/s²
CMD_ENGAGE_MOTOR = 0x04
CMD_DISENGAGE_MOTOR = 0x05
CMD_TARE_PENDULUM = 0x06
CMD_SET_TARGET = 0x07  # position-mode: commanded motor position in radians (moveTo)
CMD_ZERO_MOTOR = 0x08  # re-zero the motor step counter to the arm's current position
CMD_GET_PROFILE = 0x09
CMD_CALIBRATE_30 = 0x0A
CMD_CALIBRATE_30_REVERSE = 0x0B
CMD_GET_MOTOR_STATUS = 0x0C
CMD_GET_SAFETY = 0x0D

# time_us (uint32 LE), motor_pos, pen_pos, motor_vel, pen_vel (4× float32 LE).
_STATE_STRUCT = struct.Struct("<Iffff")
_STATE_SIZE = _STATE_STRUCT.size  # 20 bytes
_PROFILE_STRUCT = struct.Struct("<BBBI")
_MOTOR_STATUS_STRUCT = struct.Struct('<9B')


@dataclass(frozen=True)
class State:
    time_us: int           # MCU timestamp of the latest valid sensor sample
    motor_pos_rad: float   # signed motor angle, firmware frame (step counter)
    pendulum_pos_rad: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float


@dataclass(frozen=True)
class FirmwareProfile:
    protocol_version: int
    microsteps: int
    baud: int


@dataclass(frozen=True)
class MotorStatus:
    version: int
    step_pin: int
    direction_pin: int
    enable_pin: int
    pin_levels: int  # bits 0,1,2 = STEP,DIR,ENABLE; MCU pins, not driver terminals
    pin_outputs: int
    motor_engaged: bool
    planner_running: bool


class LowLevelClient:
    """Drives the LowLevelServer sketch over a serial port.

    Use as a context manager for read-only feedback:

        with LowLevelClient(port) as c:
            c.wait_until_ready()
            s = c.get_state()

    Use the guarded hardware runner for powered trials; see docs/deployment.md.
    Exiting the context attempts to disengage the motor.
    """

    def __init__(self, port: str, baud: int = 500_000, timeout: float = 0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        # Serial transactions are guarded by a lock so the client is safe to
        # use from multiple threads (e.g. an async control loop where one
        # thread does periodic get_state/set_acceleration while the main
        # thread may issue an emergency disengage_motor on signal).
        self._lock = threading.Lock()

    def __enter__(self) -> "LowLevelClient":
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Bootloader reset on serial open + setup() (which blocks on the AS5600
        # magnet) typically takes ~2 s.
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ser is not None:
            try:
                # Best-effort safe-state on exit: stop the motor.
                self.disengage_motor()
            except Exception:
                pass
            self._ser.close()
            self._ser = None

    @property
    def ser(self) -> serial.Serial:
        if self._ser is None:
            raise RuntimeError("LowLevelClient must be used as a context manager")
        return self._ser

    # --- Handshake -------------------------------------------------------

    def wait_until_ready(self, *, retries: int = 5, retry_delay_s: float = 0.5) -> bool:
        """Send READY until the board echoes it back. Returns True on success."""
        for _ in range(retries):
            with self._lock:
                self.ser.reset_input_buffer()
                self.ser.write(bytes([CMD_READY]))
                self.ser.flush()
                resp = self.ser.read(1)
            if resp == bytes([CMD_READY]):
                return True
            time.sleep(retry_delay_s)
        return False

    # --- Commands --------------------------------------------------------

    def get_state(self) -> State:
        """Request a state reading. Blocks until 20 bytes arrive or timeout."""
        with self._lock:
            self.ser.write(bytes([CMD_GET_STATE]))
            self.ser.flush()
            buf = self.ser.read(_STATE_SIZE)
        if len(buf) != _STATE_SIZE:
            raise RuntimeError(
                f"get_state: expected {_STATE_SIZE} bytes, got {len(buf)}. "
                "Is the LowLevelServer sketch flashed? Does the magnet detect?"
            )
        time_us, motor_rad, pen_rad, motor_vel, pen_vel = _STATE_STRUCT.unpack(buf)
        return State(time_us=int(time_us), motor_pos_rad=float(motor_rad),
                     pendulum_pos_rad=float(pen_rad),
                     motor_vel_rad_s=float(motor_vel),
                     pendulum_vel_rad_s=float(pen_vel))

    def get_profile(self) -> FirmwareProfile:
        """Read and validate the LowLevelServer hardware contract."""
        with self._lock:
            self.ser.write(bytes([CMD_GET_PROFILE]))
            self.ser.flush()
            buf = self.ser.read(_PROFILE_STRUCT.size)
        if len(buf) != _PROFILE_STRUCT.size:
            raise RuntimeError(
                f"get_profile: expected {_PROFILE_STRUCT.size} bytes, got {len(buf)}; "
                "re-flash the recovery LowLevelServer firmware"
            )
        command, version, microsteps, baud = _PROFILE_STRUCT.unpack(buf)
        if command != CMD_GET_PROFILE:
            raise RuntimeError(
                f"get_profile: response command 0x{command:02x}, expected "
                f"0x{CMD_GET_PROFILE:02x}"
            )
        return FirmwareProfile(int(version), int(microsteps), int(baud))

    def require_profile(
        self, *, microsteps: int, baud: int | None = None,
        protocol_version: int = 3,
    ) -> FirmwareProfile:
        """Fail closed before motion if the firmware contract disagrees."""
        expected_baud = self.baud if baud is None else int(baud)
        profile = self.get_profile()
        if (
            profile.protocol_version != int(protocol_version)
            or profile.microsteps != int(microsteps)
            or profile.baud != expected_baud
        ):
            raise RuntimeError(
                "firmware profile mismatch: "
                f"protocol={profile.protocol_version}, "
                f"microsteps={profile.microsteps}, baud={profile.baud}; "
                f"expected protocol={protocol_version}, "
                f"microsteps={microsteps}, baud={expected_baud}"
            )
        return profile

    def get_motor_status(self) -> MotorStatus:
        """Read MCU status without enabling or changing any output pin.

        This cannot measure supply voltage, motor current, or physical movement.
        """
        with self._lock:
            self.ser.write(bytes([CMD_GET_MOTOR_STATUS]))
            self.ser.flush()
            response = self.ser.read(_MOTOR_STATUS_STRUCT.size)
        if len(response) != _MOTOR_STATUS_STRUCT.size:
            raise RuntimeError('Motor-status response missing; updated firmware required')
        cmd, version, step, direction, enable, levels, outputs, engaged, running = _MOTOR_STATUS_STRUCT.unpack(response)
        if cmd != CMD_GET_MOTOR_STATUS or version != 1 or engaged > 1 or running > 1 or levels > 7 or outputs > 7:
            raise RuntimeError('Invalid motor-status response')
        return MotorStatus(version, step, direction, enable, levels, outputs, bool(engaged), bool(running))

    def get_safety_faults(self) -> int:
        """Protocol 3: latched travel=1, encoder=2, watchdog=4, bad-command=8."""
        with self._lock:
            self.ser.write(bytes([CMD_GET_SAFETY]))
            self.ser.flush()
            response = self.ser.read(3)
        if (len(response) != 3 or response[:2] != bytes([CMD_GET_SAFETY, 1])
                or response[2] & ~15):
            raise RuntimeError("Invalid firmware safety-status response")
        return response[2]

    def start_motor_calibration(self, *, direction: int = 1) -> None:
        """Start one finite +/-533-step move with fixed low firmware limits.

        Requires an enabled, idle motor near the reported center. This is
        nominally 29.98125 degrees at 6400 steps/rev, not measured arm travel.
        """
        if isinstance(direction, bool) or direction not in (-1, 1):
            raise ValueError('Calibration direction must be +1 or -1')
        command = CMD_CALIBRATE_30 if direction == 1 else CMD_CALIBRATE_30_REVERSE
        with self._lock:
            self.ser.write(bytes([command]))
            self.ser.flush()
            response = self.ser.read(2)
        if len(response) != 2 or response[0] != command:
            raise RuntimeError('Invalid calibration acknowledgement; updated firmware required')
        errors = {1: 'motor disabled', 2: 'motor busy', 3: 'motor not centered',
                  4: 'planner rejected move'}
        if response[1]:
            raise RuntimeError('Calibration rejected: ' + errors.get(response[1], 'unknown status'))

    def set_acceleration(self, accel_rad_s2: float) -> None:
        """Command a new motor angular acceleration, in rad/s².

        Firmware converts this to steps/s² for moveByAcceleration(), with
        reversal enabled and a speed cap. The write does not wait for an
        acknowledgement or confirm physical motion.
        """
        payload = bytes([CMD_SET_ACCEL]) + struct.pack("<f", float(accel_rad_s2))
        with self._lock:
            self.ser.write(payload)

    def set_target(self, target_rad: float) -> None:
        """Command a new motor position target, in radians (position mode).

        Maps to the firmware's moveTo(). Sign convention: the
        firmware frame, same as get_state and set_acceleration — one frame
        everywhere.
        The write does not wait for an acknowledgement or confirm motion.
        """
        payload = bytes([CMD_SET_TARGET]) + struct.pack("<f", float(target_rad))
        with self._lock:
            self.ser.write(payload)

    def engage_motor(self) -> None:
        with self._lock:
            self.ser.write(bytes([CMD_ENGAGE_MOTOR]))
            self.ser.flush()

    def disengage_motor(self) -> None:
        with self._lock:
            self.ser.write(bytes([CMD_DISENGAGE_MOTOR]))
            self.ser.flush()

    def tare_pendulum(self, *, timeout_s: float = 0.5) -> bool:
        """Re-zero pen_position_rad to the AS5600's current reading.

        The firmware shifts the pendulum ring buffer atomically to preserve
        velocity estimates. Returns True when the board acknowledges.
        """
        with self._lock:
            self.ser.write(bytes([CMD_TARE_PENDULUM]))
            self.ser.flush()
            resp = self.ser.read(1)
        return resp == bytes([CMD_TARE_PENDULUM])

    def zero_motor(self, *, timeout_s: float = 0.5) -> bool:
        """Re-zero the motor step counter to the arm's current position.

        Use after manually centering the arm with motor power off. The counter
        is open-loop, not an arm encoder. Returns True on acknowledgement.
        """
        with self._lock:
            self.ser.write(bytes([CMD_ZERO_MOTOR]))
            self.ser.flush()
            resp = self.ser.read(1)
        return resp == bytes([CMD_ZERO_MOTOR])
