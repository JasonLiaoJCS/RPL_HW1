#include "SafetyLimits.h"
#include <cassert>
#include <stdint.h>

int main() {
    using namespace SafetyLimits;
    static_assert(arm_steps == 1777, "100-degree limit rounded inward");
    assert(!outsideArm(1776) && !outsideArm(-1776));
    assert(outsideArm(1777) && outsideArm(-1777));
    assert(outsideArm(INT32_MAX) && outsideArm(INT32_MIN));
    assert(!commandExpired(100000, 0));
    assert(commandExpired(100001, 0));
    assert(!commandExpired(20000, UINT32_MAX-20000));
    assert(commandExpired(200000, UINT32_MAX-20000));
    assert(validDelta(65, 2000) && validDelta(-65, 2000));
    assert(!validDelta(120, 2000) && !validDelta(-120, 2000));
    assert(!validDelta(0, 0) && !validDelta(0, sample_timeout_us+1));
    assert(validDelta(0, sample_timeout_us));
}
