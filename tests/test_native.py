import numpy as np

from tether.native import speed_accel_residuals


def test_speed_accel_residual_kernel_known_constant_acceleration():
    speed = np.array([36.0, 39.6, 43.2])
    times = np.array([0.0, 0.1, 0.2])
    acceleration = np.array([10.0, 10.0, 10.0])
    residuals = speed_accel_residuals(speed, times, acceleration)
    assert np.allclose(residuals, 0.0, atol=1e-9)


def test_speed_accel_residual_kernel_handles_invalid_timestamp_and_missing_sensor():
    speed = np.array([36.0, 40.0, 43.0])
    times = np.array([0.0, 0.0, 0.2])
    acceleration = np.array([10.0, np.nan, 10.0])
    residuals = speed_accel_residuals(speed, times, acceleration)
    assert np.isnan(residuals[1])
    assert np.isnan(residuals[0])


def test_speed_accel_residual_kernel_handles_short_inputs_without_native_error():
    speed = np.array([36.0])
    times = np.array([0.0])
    acceleration = np.array([0.0])
    residuals = speed_accel_residuals(speed, times, acceleration)
    assert residuals.shape == (1,)
    assert np.isnan(residuals[0])
