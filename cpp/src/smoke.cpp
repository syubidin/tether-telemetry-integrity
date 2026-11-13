#include "tether/core.h"

#include <cmath>
#include <iostream>

int main() {
    const double speed[] = {36.0, 39.6, 43.2};
    const double time[] = {0.0, 0.1, 0.2};
    const double measured[] = {10.0, 10.0, 10.0};
    double residuals[3]{};
    const int status = tether_speed_accel_residuals(speed, time, measured, 3, residuals);
    if (status != 0 || std::abs(residuals[1]) > 1e-9) {
        std::cerr << "TETHER native kernel smoke test failed\n";
        return 1;
    }
    std::cout << "TETHER native kernel OK\n";
    return 0;
}
