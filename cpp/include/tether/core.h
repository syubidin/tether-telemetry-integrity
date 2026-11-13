#pragma once

#include <stddef.h>

#if defined(_WIN32)
  #if defined(TETHER_BUILDING_LIBRARY)
    #define TETHER_API __declspec(dllexport)
  #else
    #define TETHER_API __declspec(dllimport)
  #endif
#else
  #define TETHER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 on success, -1 for invalid pointers, -2 for an invalid length. */
TETHER_API int tether_speed_accel_residuals(
    const double* speed_kph,
    const double* timestamp_s,
    const double* measured_accel_mps2,
    size_t length,
    double* output_residuals);

#ifdef __cplusplus
}
#endif
