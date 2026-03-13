"""
Unit tests for the EKF core.

Run with:
    pytest services/ekf_service/tests/test_ekf_core.py -v

from the project root (with numpy installed).
"""

import math
import time
import numpy as np
import sys
import os

# Allow import of ekf_core and common from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from services.ekf_service.ekf_core import ExtendedKalmanFilter
from common.models import ImuMeasurement, RadarMeasurement


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_imu(t: float, ax=0.0, ay=0.0, az=0.0,
              gx=0.0, gy=0.0, gz=0.0) -> ImuMeasurement:
    return ImuMeasurement(
        timestamp=t,
        accel_x=ax, accel_y=ay, accel_z=az,
        gyro_x=gx,  gyro_y=gy,  gyro_z=gz,
    )


def _make_radar(r=100.0, az=0.0, el=0.0, dp=0.0, t=None) -> RadarMeasurement:
    return RadarMeasurement(
        timestamp=t or time.time(),
        range=r, azimuth=az, elevation=el, doppler=dp,
    )


# ── tests ────────────────────────────────────────────────────────────────────

class TestEkfPredict:

    def test_first_imu_initialises_time(self):
        """First IMU packet should set the internal timestamp but not alter state."""
        ekf = ExtendedKalmanFilter()
        x_before = ekf.x.copy()
        P_before  = ekf.P.copy()
        ekf.predict(_make_imu(0.0))
        # state unchanged on first packet
        np.testing.assert_array_equal(ekf.x, x_before)
        np.testing.assert_array_equal(ekf.P, P_before)

    def test_predict_grows_covariance(self):
        """Prediction with non-zero dt should increase covariance trace."""
        ekf = ExtendedKalmanFilter()
        trace_before = np.trace(ekf.P)
        ekf.predict(_make_imu(0.0))
        ekf.predict(_make_imu(0.1))          # dt = 0.1 s
        trace_after = np.trace(ekf.P)
        assert trace_after > trace_before, \
            "Predict should grow covariance (process noise added)"

    def test_predict_constant_velocity(self):
        """With zero accel IMU, position should advance by v*dt."""
        ekf = ExtendedKalmanFilter()
        # set an initial velocity
        ekf.x[3] = 5.0   # vx = 5 m/s
        ekf.predict(_make_imu(0.0))
        dt = 0.1
        ekf.predict(_make_imu(dt))
        # position should increase by approx 5 * 0.1 = 0.5  (x[0] started at 100)
        assert abs(ekf.x[0] - (100.0 + 5.0 * dt)) < 0.01, \
            f"Expected px ≈ {100.0 + 5.0*dt}, got {ekf.x[0]}"

    def test_quaternion_stays_unit(self):
        """After several predict steps the quaternion must remain unit."""
        ekf = ExtendedKalmanFilter()
        t = 0.0
        dt = 0.01
        ekf.predict(_make_imu(t, gx=0.1, gy=0.05, gz=-0.02))
        for _ in range(200):
            t += dt
            ekf.predict(_make_imu(t, gx=0.1, gy=0.05, gz=-0.02))
        qnorm = np.linalg.norm(ekf.x[6:10])
        assert abs(qnorm - 1.0) < 1e-6, \
            f"Quaternion norm drifted to {qnorm}"

    def test_predict_ignores_stale_packet(self):
        """A packet with negative dt should not change state."""
        ekf = ExtendedKalmanFilter()
        ekf.predict(_make_imu(10.0))
        x_snap = ekf.x.copy()
        ekf.predict(_make_imu(9.0))   # stale — dt < 0
        np.testing.assert_array_equal(ekf.x, x_snap)


class TestEkfUpdate:

    def test_update_reduces_covariance(self):
        """A radar update should reduce (or maintain) covariance trace."""
        ekf = ExtendedKalmanFilter()
        # prime the time reference
        ekf.predict(_make_imu(0.0))
        ekf.predict(_make_imu(0.1))

        trace_before = np.trace(ekf.P)
        # Measurement consistent with the prior (target at ~100 m ahead)
        ekf.update(_make_radar(r=100.0, az=0.0, el=0.0, dp=3.0))
        trace_after = np.trace(ekf.P)
        assert trace_after < trace_before, \
            "Update should reduce covariance (information gained)"

    def test_update_moves_state_toward_measurement(self):
        """After a radar update with range=50, EKF position magnitude should move toward 50 m."""
        ekf = ExtendedKalmanFilter()
        ekf.predict(_make_imu(0.0))
        ekf.predict(_make_imu(0.1))

        r_before = np.linalg.norm(ekf.x[0:3])
        # Force a measurement well away from the prior
        ekf.update(_make_radar(r=50.0, az=0.0, el=0.0, dp=0.0))
        r_after = np.linalg.norm(ekf.x[0:3])
        # range should move toward 50 m from about 100 m
        assert r_after < r_before, \
            "State should move toward the measurement (range = 50 m)"

    def test_covariance_positive_definite_after_update(self):
        """Covariance must remain positive semi-definite after update."""
        ekf = ExtendedKalmanFilter()
        ekf.predict(_make_imu(0.0))
        ekf.predict(_make_imu(0.1))
        ekf.update(_make_radar(r=99.0, az=0.01, el=-0.01, dp=2.5))
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues >= -1e-9), \
            f"Covariance has negative eigenvalue: {eigenvalues.min()}"

    def test_get_state_matches_internal(self):
        """get_state() should faithfully mirror the internal state vector."""
        ekf = ExtendedKalmanFilter()
        ekf.predict(_make_imu(0.0))
        ekf.predict(_make_imu(0.05))
        ekf.update(_make_radar(r=98.0))
        s = ekf.get_state(timestamp=1.0)
        assert s.pos_x == ekf.x[0]
        assert s.qw   == ekf.x[6]
        assert len(s.covariance) == 100

    def test_repeated_predict_update_cycle(self):
        """EKF should remain numerically stable over 100 predict+update cycles."""
        ekf = ExtendedKalmanFilter()
        t = 0.0
        dt_imu   = 0.01
        dt_radar = 0.1
        t_radar  = dt_radar
        ekf.predict(_make_imu(t))

        for i in range(1000):
            t += dt_imu
            ekf.predict(_make_imu(t))
            if t >= t_radar:
                ekf.update(_make_radar(r=95.0 - i*0.005, az=0.0, el=0.0, dp=3.0))
                t_radar += dt_radar

        assert np.all(np.isfinite(ekf.x)), "State contains NaN/Inf"
        assert np.all(np.isfinite(ekf.P)), "Covariance contains NaN/Inf"
