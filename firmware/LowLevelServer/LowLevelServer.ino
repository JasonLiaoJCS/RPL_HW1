#include <FastAccelStepper.h>
#include <AS5600.h>
#include <Wire.h>
#include "SafetyLimits.h"

// Lab profile: DRV8825 at 1/32 and 500000 baud. PlatformIO supplies matching
// build flags; these defaults also support Arduino CLI builds.
#ifndef RIP_SERIAL_BAUD
#define RIP_SERIAL_BAUD 500000L
#endif
#ifndef RIP_MICROSTEPS
#define RIP_MICROSTEPS 32
#endif
const long BAUD_RATE = RIP_SERIAL_BAUD;

// Command bytes
#define CMD_READY 0x01
#define CMD_GET_STATE 0x02
#define CMD_SET_ACCEL 0x03   // commanded angular acceleration (rad/s²)
#define CMD_ENGAGE_MOTOR 0x04
#define CMD_DISENGAGE_MOTOR 0x05
#define CMD_TARE_PENDULUM 0x06   // re-zero pen_position_rad to current AS5600 reading
#define CMD_SET_TARGET 0x07      // position-mode: commanded motor position (rad) → moveTo
#define CMD_ZERO_MOTOR 0x08      // re-zero the motor step counter to the arm's current position
#define CMD_GET_PROFILE 0x09     // firmware contract: version, microsteps, baud
#define CMD_CALIBRATE_30 0x0A    // finite +533 microsteps; reply command,status
#define CMD_CALIBRATE_30_REVERSE 0x0B // finite -533 microsteps; same limits
#define CMD_GET_SAFETY 0x0D // command, version=1, latched fault bitmask
#define CMD_GET_MOTOR_STATUS 0x0C // read-only MCU pin configuration and planner state

// Pin assignments. STEP must be on pin 9 (Timer1 OC1A on ATmega328) for
// FastAccelStepper. DIR and ENABLE can be any digital pin.
#define DIR_PIN 2
#define STEP_PIN 9
#define ENABLE_PIN 5

// Lab contract: DRV8825 1/32 with MODE0, MODE1 and MODE2 high.
// Must match the physical driver settings and MOTOR_MICROSTEPS in the simulator.
const int MICROSTEPS = RIP_MICROSTEPS;
#if RIP_MICROSTEPS != 32
#error "Recovery firmware requires DRV8825 1/32 microstepping"
#endif
const long STEPS_PER_REVOLUTION = 200L * MICROSTEPS;
const float STEPS_PER_RAD = (float)STEPS_PER_REVOLUTION / (2.0f * (float)PI);
const float RAD_PER_STEP = (2.0f * (float)PI) / (float)STEPS_PER_REVOLUTION;

// Planner speed cap: approximately 5 rad/s at the configured microstep ratio.
// Applies to acceleration and position commands. Match the simulator's
// MOTOR_PLANNER_SPEED_LIMIT_RAD_S, not its optional velocity-action limit.
const uint32_t MOTOR_MIN_STEP_US =  // ≈ 5 rad/s, derived so it tracks MICROSTEPS
    (uint32_t)(1.0e6f * 2.0f * (float)PI / (5.0f * (float)STEPS_PER_REVOLUTION));

// Ramp acceleration for position-mode moveTo() (CMD_SET_TARGET).
// Must be re-asserted on every CMD_SET_TARGET: CMD_ENGAGE_MOTOR's
// moveByAcceleration(0) resets the library acceleration to 1 step/s², which
// would make the motor merely creep toward the target. Accel mode ignores this.
const int32_t MOTOR_ACCEL_STEPS_S2 = 50000;

const int32_t MOTOR_SAFE_LIMIT_STEPS = SafetyLimits::arm_steps;

// Encoder samples are kept in a ring buffer updated at 500 Hz; GET_STATE
// returns velocity computed as (newest - oldest)/Δt over a window of 5
// samples = 4 inter-sample gaps = 8 ms. A shorter window reduces observation
// lag at the cost of a noisier per-sample velocity estimate.
const uint16_t SAMPLE_PERIOD_US = 2000;
const uint8_t  BUFFER_SIZE      = 16;
const uint8_t  VEL_WINDOW       = 5;
static int32_t motor_step_buf[BUFFER_SIZE];   // raw stepper position (steps)
static float   pen_rad_buf[BUFFER_SIZE];      // accumulated pendulum angle (rad)
static uint32_t time_us_buf[BUFFER_SIZE];     // sample timestamps (µs)
static uint8_t  buf_head = 0;                 // next write index
static bool     buf_filled = false;           // becomes true after first full lap
static uint32_t last_sample_us = 0;
static uint32_t last_valid_sample_us = 0;
static uint8_t safety_fault = 0;

// Continuously-tracked pendulum angle (independent of GET_STATE cadence so
// rapid wraparounds aren't missed).
static long    pen_raw_prev   = -1;           // -1 = first read sentinel
static float   pen_position_rad = 0.0f;

// State variables
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5600 as5600;

// `motor_engaged` is only touched from loop() / handleCommand() — no ISR
// access — so `volatile` would only mislead future readers. Plain bool.
bool motor_engaged = false;
// Homework release: host acceleration commands must arrive continuously.
// Finite position/calibration moves are bounded and use their own completion.
bool command_watchdog = false;
uint32_t last_command_us = 0;
void latchFault(uint8_t reason)
{
    safety_fault |= reason;
    motor_engaged = false;
    command_watchdog = false;
    stepper->forceStopAndNewPosition(stepper->getCurrentPosition());
    stepper->disableOutputs();
}

uint8_t readMotorPinMask(bool output_modes)
{
    const uint8_t pins[] = {STEP_PIN, DIR_PIN, ENABLE_PIN};
    uint8_t mask = 0;
    for (uint8_t i = 0; i < 3; i++)
    {
        const uint8_t port = digitalPinToPort(pins[i]);
        // Do not use digitalRead(STEP_PIN): Arduino's implementation disables
        // timer output on PWM-capable pins, including our OC1A step output.
        const uint8_t value = output_modes ? *portModeRegister(port) : *portInputRegister(port);
        if (value & digitalPinToBitMask(pins[i])) mask |= (1 << i);
    }
    return mask;
}

// Function prototypes
void handleCommand();
void sendState();
void sampleState();
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s);

void setup()
{
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, HIGH);
    Serial.begin(BAUD_RATE);
    Wire.begin();
    Wire.setWireTimeout(SafetyLimits::i2c_timeout_us, true);
    Wire.setClock(400000);   // I²C fast mode for short transaction times.
    as5600.begin();

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        while (true) { /* halt: STEP_PIN is not Timer1 OC1A/OC1B */ }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN, true); // explicitly active-low
    stepper->setAutoEnable(false);

    int8_t rc_speed = stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
    if (rc_speed != 0)
    {
        while (true) {}
    }
    // Acceleration is set per tick in CMD_SET_ACCEL.
    // Use an 8 ms planning window to limit command latency. Already queued
    // step pulses cannot be changed by a new acceleration command.
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    while (!Serial) { ; }

    last_sample_us = micros();
}

void loop()
{
    uint32_t now_us = micros();
    // Enforce locally, even when the host stops sending commands.
    if (motor_engaged && SafetyLimits::outsideArm(stepper->getCurrentPosition()))
        latchFault(SafetyLimits::travel);
    if (motor_engaged && command_watchdog && SafetyLimits::commandExpired(now_us, last_command_us))
        latchFault(SafetyLimits::command_timeout);
    if ((uint32_t)(now_us - last_sample_us) >= SAMPLE_PERIOD_US)
    {
        last_sample_us = now_us;
        sampleState();
    }
    if (Serial.available() > 0)
    {
        handleCommand();
    }
}

/*
 * Append a (time, motor_position, pendulum_position) sample to the ring
 * buffer and update the pendulum wraparound accumulator.
 */
void sampleState()
{
    if (safety_fault) return;
    const bool magnet = as5600.magnetDetected();
    const int status_error = as5600.lastError();
    const long raw = as5600.rawAngle();
    const int angle_error = as5600.lastError();
    if (!magnet || status_error || angle_error || Wire.getWireTimeoutFlag())
    {
        latchFault(SafetyLimits::encoder);
        return; // Never stamp an old/failed encoder value as a fresh sample.
    }
    int32_t motor_step = stepper->getCurrentPosition();
    if (pen_raw_prev < 0)
    {
        pen_raw_prev = raw;
    }
    else
    {
        long delta = raw - pen_raw_prev;
        if (delta > 2048) delta -= 4096;
        if (delta < -2048) delta += 4096;
        if (!SafetyLimits::validDelta(delta, (uint32_t)(last_sample_us-last_valid_sample_us)))
        {
            latchFault(SafetyLimits::encoder);
            return;
        }
        pen_position_rad += (float)delta * ((2.0f * PI) / 4096.0f);
        pen_raw_prev = raw;
    }
    last_valid_sample_us = last_sample_us;

    motor_step_buf[buf_head] = motor_step;
    pen_rad_buf[buf_head]    = pen_position_rad;
    time_us_buf[buf_head]    = last_sample_us;
    buf_head = (buf_head + 1) % BUFFER_SIZE;
    if (buf_head == 0) buf_filled = true;
}

/*
 * Compute velocity over the most recent VEL_WINDOW samples as
 * (newest - oldest)/Δt. Returns 0 until the buffer holds enough samples.
 */
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s)
{
    uint8_t n_samples = buf_filled ? BUFFER_SIZE : buf_head;
    if (n_samples < VEL_WINDOW)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1)            % BUFFER_SIZE);
    uint8_t oldest = (uint8_t)((buf_head + BUFFER_SIZE - VEL_WINDOW)   % BUFFER_SIZE);

    uint32_t t_new = time_us_buf[newest];
    uint32_t t_old = time_us_buf[oldest];
    float dt_s = (float)((uint32_t)(t_new - t_old)) * 1e-6f;
    if (dt_s <= 0.0f)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    int32_t motor_step_delta = motor_step_buf[newest] - motor_step_buf[oldest];
    *motor_vel_rad_s = ((float)motor_step_delta * ((2.0f * PI) / (float)STEPS_PER_REVOLUTION)) / dt_s;

    float pen_delta = pen_rad_buf[newest] - pen_rad_buf[oldest];
    *pen_vel_rad_s = pen_delta / dt_s;
}

/*
 * Handle incoming commands.
 */
void handleCommand()
{
    uint8_t command = Serial.read();

    switch (command)
    {
    case CMD_READY:
        Serial.write(CMD_READY);
        break;

    case CMD_GET_STATE:
        sendState();
        break;

    case CMD_SET_ACCEL:
        {
            // Read 4 bytes (size of float) with a short timeout. If a byte
            // gets dropped — UART overrun under stepper EMI, or a flipped
            // command byte that desyncs the parser — readBytes returns
            // short and the handler latches a fault.
            Serial.setTimeout(5);
            float accel_rad_s2;
            size_t n = Serial.readBytes((char *)&accel_rad_s2, sizeof(float));
            Serial.setTimeout(1000);  // restore Stream default
            if (n != sizeof(float)) { latchFault(SafetyLimits::invalid_command); break; }

            if (!motor_engaged) break;

            if (!isfinite(accel_rad_s2) || fabs(accel_rad_s2) > 50.0f)
            {
                latchFault(SafetyLimits::invalid_command);
                break;
            }
            command_watchdog = true;
            last_command_us = micros();

            // Convert rad/s² to steps/s² (STEPS_PER_REVOLUTION microsteps per revolution).
            // moveByAcceleration takes int32_t.
            int32_t accel_steps_s2 =
                (int32_t)(accel_rad_s2 * ((float)STEPS_PER_REVOLUTION / (2.0f * PI)));

            if (SafetyLimits::outsideArm(stepper->getCurrentPosition()))
            {
                latchFault(SafetyLimits::travel);
                break;
            }

            // allow_reverse=true makes the library decelerate smoothly through
            // zero when the sign of accel opposes the current velocity — no
            // state machine needed on our side.
            stepper->moveByAcceleration(accel_steps_s2, true);
        }
        break;

    case CMD_SET_TARGET:
        {
            // Position-mode command: absolute motor target in radians. The host
            // integrates the policy's per-tick deltas and sends the absolute
            // target each tick; the stepper ramps toward it (moveTo). Coexists
            // with CMD_SET_ACCEL. Targets and GET_STATE use the same sign.
            Serial.setTimeout(5);
            float target_rad;
            size_t n = Serial.readBytes((char *)&target_rad, sizeof(float));
            Serial.setTimeout(1000);  // restore Stream default
            if (n != sizeof(float)) { latchFault(SafetyLimits::invalid_command); break; }

            if (!motor_engaged) break;

            if (!isfinite(target_rad) || fabs(target_rad) >= (100.0f * PI / 180.0f))
            {
                latchFault(SafetyLimits::invalid_command);
                break;
            }
            // Bound the step-count target; this is not physical angle feedback.
            int32_t target_steps = (int32_t)lroundf(target_rad * STEPS_PER_RAD);
            if (target_steps >  MOTOR_SAFE_LIMIT_STEPS) target_steps =  MOTOR_SAFE_LIMIT_STEPS;
            if (target_steps < -MOTOR_SAFE_LIMIT_STEPS) target_steps = -MOTOR_SAFE_LIMIT_STEPS;

            // Re-assert the ramp acceleration (CMD_ENGAGE_MOTOR left it at
            // 1 step/s²; see the constant's note).
            stepper->setAcceleration(MOTOR_ACCEL_STEPS_S2);
            command_watchdog = false;
            stepper->moveTo(target_steps);
        }
        break;

    case CMD_ENGAGE_MOTOR:
        if (safety_fault || pen_raw_prev < 0 ||
            (uint32_t)(micros()-last_valid_sample_us) > SafetyLimits::sample_timeout_us)
            break;
        if (SafetyLimits::outsideArm(stepper->getCurrentPosition()))
        {
            latchFault(SafetyLimits::travel);
            break;
        }
        last_command_us = micros();
        command_watchdog = true;
        // Restore the normal speed rail after a finite calibration move.
        stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
        motor_engaged = stepper->enableOutputs();
        if (!motor_engaged || (readMotorPinMask(false) & 0x04))
        {
            motor_engaged = false;
            stepper->disableOutputs();
            break;
        }
        // Start in zero-accel state. moveByAcceleration(0, true) means
        // "hold current speed", and since we've just enabled the driver
        // the stepper is at rest — so this leaves it at rest until the
        // host sends its first CMD_SET_ACCEL.
        // Note: ENGAGE does not re-zero `getCurrentPosition()` or
        // `pen_position_rad`. Position counters persist across
        // engage/disengage cycles (and across episodes) by design — the
        // host is responsible for tracking the frame.
        stepper->moveByAcceleration(0, true);
        break;

    case CMD_DISENGAGE_MOTOR:
        motor_engaged = false;
        // forceStop() lets queued pulses finish. Abort that queue before
        // disabling the driver so disabled pulses cannot advance the counter.
        // Preserve a best-effort step-count snapshot, not a measured arm angle.
        stepper->forceStopAndNewPosition(stepper->getCurrentPosition());
        stepper->disableOutputs();
        break;

    case CMD_TARE_PENDULUM:
        if (motor_engaged || safety_fault) break;
        // Re-zero pen_position_rad to the current AS5600 reading.
        //
        // Shift pen_position_rad AND every entry in pen_rad_buf by the
        // current pen_position_rad. The buffer's relative deltas (used
        // by computeVelocities) are preserved, so velocity calculation
        // continues uninterrupted across the tare. Disable interrupts so
        // sampleState() doesn't run mid-update with a partially-shifted
        // buffer.
        noInterrupts();
        {
            float offset = pen_position_rad;
            for (uint8_t i = 0; i < BUFFER_SIZE; i++)
            {
                pen_rad_buf[i] -= offset;
            }
            pen_position_rad = 0.0f;
        }
        interrupts();
        Serial.write(CMD_TARE_PENDULUM);  // ack
        break;

    case CMD_ZERO_MOTOR:
        if (motor_engaged || safety_fault) break;
        // Re-zero the step counter to the arm's current position, for when the
        // operator has moved the arm by hand between episodes. Shift the ring
        // buffer by the same offset (as CMD_TARE_PENDULUM does) so velocity
        // stays continuous; interrupts off for a consistent buffer.
        noInterrupts();
        {
            int32_t offset = stepper->getCurrentPosition();
            for (uint8_t i = 0; i < BUFFER_SIZE; i++)
            {
                motor_step_buf[i] -= offset;
            }
            stepper->setCurrentPosition(0);
        }
        interrupts();
        Serial.write(CMD_ZERO_MOTOR);  // ack
        break;

    case CMD_CALIBRATE_30:
    case CMD_CALIBRATE_30_REVERSE:
        {
            uint8_t status = 0;
            const int32_t origin = stepper->getCurrentPosition();
            if (!motor_engaged) status = 1;
            else if (stepper->isRunning()) status = 2;
            else if (origin < -88 || origin > 88) status = 3; // about +/-5 deg
            else if (stepper->setSpeedInHz(256) != 0 ||
                     stepper->setAcceleration(512) != 0) status = 4;
            else if (stepper->move(command == CMD_CALIBRATE_30 ? 533 : -533) != MOVE_OK) status = 4;
            if (status == 0) command_watchdog = false;
            const uint8_t response[] = {command, status};
            Serial.write(response, sizeof(response));
        }
        break;

    case CMD_GET_SAFETY:
        {
            const uint8_t response[] = {CMD_GET_SAFETY, 1, safety_fault};
            Serial.write(response, sizeof(response));
        }
        break;

    case CMD_GET_MOTOR_STATUS:
        {
            const uint8_t response[] = {
                CMD_GET_MOTOR_STATUS, 1, STEP_PIN, DIR_PIN, ENABLE_PIN,
                readMotorPinMask(false), readMotorPinMask(true),
                (uint8_t)motor_engaged, (uint8_t)stepper->isRunning()
            };
            Serial.write(response, sizeof(response));
        }
        break;

    case CMD_GET_PROFILE:
        // Little-endian packed response: command, protocol version,
        // microsteps, baud. The host refuses to drive a firmware profile that
        // disagrees with its policy/simulation contract.
        {
            const uint8_t response[] = {
                CMD_GET_PROFILE,
                3,
                (uint8_t)MICROSTEPS,
                (uint8_t)(BAUD_RATE & 0xff),
                (uint8_t)((BAUD_RATE >> 8) & 0xff),
                (uint8_t)((BAUD_RATE >> 16) & 0xff),
                (uint8_t)((BAUD_RATE >> 24) & 0xff),
            };
            Serial.write(response, sizeof(response));
        }
        break;

    default:
        if (motor_engaged) latchFault(SafetyLimits::invalid_command);
        break;
    }
}

/*
 * Send the current state of the system:
 * - Current time in microseconds   (4 bytes, uint32)
 * - Stepper motor position in rad  (4 bytes, float)
 * - Pendulum position in rad       (4 bytes, float)
 * - Stepper motor velocity in rad/s (4 bytes, float)
 * - Pendulum velocity in rad/s     (4 bytes, float)
 */
void sendState()
{
    // Take the most recent buffer entry as the instantaneous position,
    // and compute velocity from the (newest - oldest)/Δt finite-difference
    // window. The timestamp is the sample time of `newest` — not a
    // fresh micros() — so the host gets a self-consistent (t, pos, vel)
    // tuple it can time-align without inheriting up-to-one-sample of
    // bias.
    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1) % BUFFER_SIZE);
    uint32_t current_time = time_us_buf[newest];
    float motor_position_radians = (float)motor_step_buf[newest] * RAD_PER_STEP;
    float pendulum_position_radians = pen_rad_buf[newest];

    float motor_velocity_rad_s, pendulum_velocity_rad_s;
    computeVelocities(&motor_velocity_rad_s, &pendulum_velocity_rad_s);
    if (safety_fault || pen_raw_prev < 0) pendulum_position_radians = NAN;

    // Values go out exactly as the firmware measures them (step counter,
    // AS5600 accumulator) — one frame everywhere, no flips on either side.

    // Pack and send the data
    Serial.write((byte *)&current_time, sizeof(current_time));
    Serial.write((byte *)&motor_position_radians, sizeof(motor_position_radians));
    Serial.write((byte *)&pendulum_position_radians, sizeof(pendulum_position_radians));
    Serial.write((byte *)&motor_velocity_rad_s, sizeof(motor_velocity_rad_s));
    Serial.write((byte *)&pendulum_velocity_rad_s, sizeof(pendulum_velocity_rad_s));
}
