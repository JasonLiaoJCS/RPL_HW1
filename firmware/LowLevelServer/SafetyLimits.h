#pragma once
#include <stdint.h>
#include <math.h>

// Step-count protection is not an independent physical arm measurement.
namespace SafetyLimits {
constexpr int32_t arm_steps = 6400L * 100L / 360L;
constexpr uint32_t command_timeout_us = 100000;
constexpr uint32_t sample_timeout_us = 10000;
constexpr uint32_t i2c_timeout_us = 1500;
enum Fault : uint8_t { travel = 1, encoder = 2, command_timeout = 4, invalid_command = 8 };

inline bool outsideArm(int32_t position) {
    return position <= -arm_steps || position >= arm_steps;
}
inline bool commandExpired(uint32_t now, uint32_t last) {
    return (uint32_t)(now-last) > command_timeout_us;
}
inline bool validDelta(long delta, uint32_t elapsed) {
    if (elapsed == 0 || elapsed > sample_timeout_us) return false;
    const float limit = 50.0f * ((float)elapsed * 1e-6f) * (4096.0f / 6.283185307f) + 4.0f;
    return fabsf((float)delta) <= limit;
}
}
