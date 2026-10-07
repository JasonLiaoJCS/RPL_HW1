# Assemble the Rig

This release targets an ATmega328P Arduino Nano, a 200-full-step/revolution NEMA17
motor, a **DRV8825 at 1/32 microstepping**, and an AS5600 pendulum encoder.
There is no physical arm encoder. A firmware step count can advance even when
the motor stalls; inspect actual movement during commissioning.

## Parts and mechanical assembly

Use the course kit: Nano/USB cable, DRV8825 carrier with cooling, NEMA17 motor,
AS5600 breakout and diametrically magnetized magnet, switched 12 V supply,
bearing (608, 8 x 22 x 7 mm), wire/connectors, protoboard and printed parts.
The five STLs are in `hardware/meshes`: base, lid, motor plate, arm and pendulum.
The arm length is 65 mm from motor axis to pendulum axis.

1. Slice and print the parts, inspect fits, and mount the motor securely in the
   motor plate and base. Use the kit's supplied mounting hardware. Verify the
   actual motor screw depth and print fit; fastener lengths are not specified
   by the CAD files.
2. The kit pendulum is printed with an NT$10 coin (7.5 g) sealed in its pocket
   as the tip mass. The CAD URDF assumes a 7.12 g coin, so
   weigh your complete pendulum and record it in your parameter file (see
   [system identification](system-identification.md#prepare)). If you reprint
   the pendulum, preview the sliced pocket and pause to insert the coin before
   it closes; do not blindly copy a layer number between slicer settings. The
   supplied parameter file contains synthetic starter values, not a calibration
   of an assembled rig.
3. Fit the arm securely to the motor shaft. Install the bearing and pendulum at
   the arm tip; check free swing and retention without binding.
4. Center the magnet on the pendulum axis facing the AS5600. Secure the sensor
   and magnet so the gap cannot change during motion. Verify magnet detection
   with the encoder sketch before final assembly.
5. Route wires with slack throughout the intended +/-100-degree arm envelope.
   Mark physical arm center on the base. Check free travel with power off.
6. Weigh/measure your pendulum, note differences from the example, and record a
   free swing. Copy and adjust `plant/data/example_rig.json` for your own rig;
   pass it via `--params`. Identify and validate your own model instead of
   treating the supplied values as measurements.

Component images in `hardware/diagrams` are identification aids. Pin labels and
the table below govern this release. Do not infer wiring, microstep settings or
baud rates from the component images alone.

## Wiring

Disconnect 12 V and USB while wiring. Identify motor coil pairs with a meter;
wire each complete coil to one driver output pair, independent of wire color.

| Connection | Course rig |
| --- | --- |
| Nano D9 | DRV8825 STEP (timer output; do not move to a random digital pin) |
| Nano D2 | DRV8825 DIR |
| Nano D5 | DRV8825 ENABLE, active low |
| Nano A4 / A5 | AS5600 SDA / SCL |
| Nano logic supply / GND | AS5600 power and ground, according to breakout rating |
| Driver GND | Common ground with Nano and 12 V supply |
| Switched +12 V | Driver VMOT and Nano VIN; keep it off the Nano 5 V pin and sensor pins |
| Driver A1/A2, B1/B2 | Two identified stepper coils |
| Driver MODE0/MODE1/MODE2 | All high for 1/32 on the DRV8825 |
| Driver RESET and SLEEP | Logic high, following the carrier schematic |

Solder the kit's 100 nF ceramic and 22 uF electrolytic capacitors across
VMOT/GND close to the driver, with the electrolytic's polarity correct. Before
the motor ever runs, set the current limit to Vref = 0.45 V: about 0.9 A peak,
since Vref = Imax/2 with the usual 0.1 ohm sense resistors. Measure Vref against
GND with the driver powered and the motor disconnected. Never unplug the motor
with the driver powered. The
[DRV8825 carrier documentation](https://www.pololu.com/product/2133) describes
the current-limit procedure.
Confirm Nano pin labels against the
[Arduino Nano documentation](https://docs.arduino.cc/hardware/nano/).

## First checks

USB powers the controller while the switched motor rail stays off. Flash
heartbeat, then encoder diagnostics, then the release low-level server as
described in [deployment](deployment.md). Turn the pendulum by hand with motor
power off and verify continuous angle tracking and correct hanging zero.

Before a powered run, have the TA inspect the assembly, coil pairing, driver
current setting, center mark and wire clearance. After the finite motor check,
compare reported and physical +30-degree motion. Buzzing with no motion, wrong
angle, a loose shaft or poor holding torque must be resolved before policy tests.
Rewriting controller gains does not correct a wrong microstep setting.
