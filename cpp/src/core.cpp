#include "tether/core.h"

#include <cmath>
#include <limits>

extern "C" int tether_speed_accel_residuals(
    const double* speed_kph,
    const double* timestamp_s,
    const double* measured_accel_mps2,
    const size_t length,
    double* output_residuals) {
    if (speed_kph == nullptr || timestamp_s == nullptr || measured_accel_mps2 == nullptr || output_residuals == nullptr) {
        return -1;
    }
    if (length < 2 || length > 1000000) {
        return -2;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (size_t index = 0; index < length; ++index) {
        output_residuals[index] = nan;
    }

    for (size_t index = 0; index < length; ++index) {
        size_t left = index == 0 ? 0 : index - 1;
        size_t right = index + 1 < length ? index + 1 : index;
        const double dt = timestamp_s[right] - timestamp_s[left];
        if (!std::isfinite(speed_kph[left]) || !std::isfinite(speed_kph[right]) ||
            !std::isfinite(timestamp_s[left]) || !std::isfinite(timestamp_s[right]) ||
            !std::isfinite(measured_accel_mps2[index]) || dt <= 0.0) {
            continue;
        }
        const double speed_left = speed_kph[left] / 3.6;
        const double speed_right = speed_kph[right] / 3.6;
        const double derived_accel = (speed_right - speed_left) / dt;
        output_residuals[index] = derived_accel - measured_accel_mps2[index];
    }
    return 0;
}
