"""
EKF Core — Extended Kalman Filter for Radar-IMU Sensor Fusion
=============================================================

State vector  x ∈ ℝ¹⁰:
    [0:3]  position  p  (m)      in world/NED frame
    [3:6]  velocity  v  (m/s)    in world frame
    [6:10] quaternion q  (w,x,y,z) — body orientation

Inputs
------
Predict : ImuMeasurement   — body-frame accel + gyro
Update  : RadarMeasurement — spherical (range, azimuth, elevation, doppler)

References
----------
Groves, P.D. (2013). Principles of GNSS, Inertial, and Multisensor
Integrated Navigation Systems (Chapter 3, 14).
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from common.models import ImuMeasurement, RadarMeasurement, EkfState

FloatArray = NDArray[np.float64]

# ── helpers ─────────────────────────────────────────────────────────────────

def _skew(v: FloatArray) -> FloatArray:
    """3×3 skew-symmetric matrix of vector *v*."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])


def _quat_norm(q: FloatArray) -> FloatArray:
    """Normalise a quaternion in-place and return it."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_to_rot(q: FloatArray) -> FloatArray:
    """Convert unit quaternion (w,x,y,z) → 3×3 rotation matrix R.

    Transforms **body** vectors into **world** frame: v_w = R @ v_b
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)    ],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)    ],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def _omega_matrix(gyro: FloatArray) -> FloatArray:
    """4×4 Omega matrix for quaternion kinematics: dq/dt = 0.5 Ω(ω) q."""
    gx, gy, gz = gyro
    return 0.5 * np.array([
        [ 0,  -gx, -gy, -gz],
        [ gx,  0,   gz, -gy],
        [ gy, -gz,  0,   gx],
        [ gz,  gy, -gx,  0  ],
    ])


# ── EKF class ───────────────────────────────────────────────────────────────

class ExtendedKalmanFilter:
    """EKF fusing IMU (predict) and radar (update) measurements.

    Parameters
    ----------
    sigma_accel    : accelerometer noise std-dev (m/s²)
    sigma_gyro     : gyroscope noise std-dev (rad/s)
    sigma_range    : radar range noise std-dev (m)
    sigma_angle    : radar azimuth/elevation noise std-dev (rad)
    sigma_doppler  : radar Doppler noise std-dev (m/s)
    """

    N = 10  # state dimension

    def __init__(
        self,
        sigma_accel:   float = 0.1,
        sigma_gyro:    float = 0.01,
        sigma_range:   float = 2.0,
        sigma_angle:   float = 0.02,
        sigma_doppler: float = 0.5,
    ) -> None:
        # ── state ──────────────────────────────────────────────────────────
        self.x: FloatArray = np.zeros(self.N)
        self.x[0] = 100.0   # initial position guess (m) — coarse
        self.x[6] = 1.0     # quaternion w = 1  (identity rotation)

        # ── covariance ─────────────────────────────────────────────────────
        self.P: FloatArray = np.diag([
            50**2, 50**2, 25**2,   # position uncertainty (m²)
            10**2, 10**2,  5**2,   # velocity uncertainty (m/s)²
            0.1**2, 0.1**2, 0.1**2, 0.1**2,  # quaternion
        ])

        # ── process noise (Q) ──────────────────────────────────────────────
        qa = sigma_accel**2
        qg = sigma_gyro**2
        self._Q_accel = qa
        self._Q_gyro  = qg

        # ── measurement noise (R) ──────────────────────────────────────────
        self.R: FloatArray = np.diag([
            sigma_range**2,
            sigma_angle**2,
            sigma_angle**2,
            sigma_doppler**2,
        ])

        self._last_time: float | None = None

    # ------------------------------------------------------------------
    # Predict step — driven by IMU
    # ------------------------------------------------------------------

    def predict(self, imu: ImuMeasurement) -> None:
        """Propagate state forward using IMU body-frame measurements."""
        if self._last_time is None:
            self._last_time = imu.timestamp
            return

        dt = imu.timestamp - self._last_time
        if dt <= 0.0 or dt > 1.0:          # ignore stale / future packets
            self._last_time = imu.timestamp
            return
        self._last_time = imu.timestamp

        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]

        a_body = np.array([imu.accel_x, imu.accel_y, imu.accel_z])
        w_body = np.array([imu.gyro_x,  imu.gyro_y,  imu.gyro_z ])

        R = _quat_to_rot(q)
        a_world = R @ a_body          # rotate acceleration into world frame

        # ── state propagation (Euler integration) ─────────────────────────
        p_new = p + v * dt + 0.5 * a_world * dt**2
        v_new = v + a_world * dt
        Omega = _omega_matrix(w_body)
        q_new = _quat_norm(q + (Omega @ q) * dt)

        self.x[0:3]  = p_new
        self.x[3:6]  = v_new
        self.x[6:10] = q_new

        # ── Jacobian F of f(x) w.r.t. x  (10×10) ─────────────────────────
        F = np.eye(self.N)

        # ∂p/∂v
        F[0:3, 3:6] = np.eye(3) * dt

        # ∂v/∂q  — full closed-form (Sola 2017, Appendix E):
        # dR(q)/dq ⊗ a  is a 3×4 matrix G(q,a)
        w_, x_, y_, z_ = q
        a = a_body
        G = 2.0 * np.column_stack([
            # w column
            np.cross(np.array([x_, y_, z_]), a),
            # x column
            np.array([x_*a[0]+y_*a[1]+z_*a[2], y_*a[0]-x_*a[1]-w_*a[2], z_*a[0]+w_*a[1]-x_*a[2]]),
            # y column
            np.array([-y_*a[0]+x_*a[1]+w_*a[2], x_*a[0]+y_*a[1]+z_*a[2], -w_*a[0]+z_*a[1]-y_*a[2]]),
            # z column
            np.array([-z_*a[0]-w_*a[1]+x_*a[2], w_*a[0]-z_*a[1]+y_*a[2], x_*a[0]+y_*a[1]+z_*a[2]]),
        ])
        F[3:6, 6:10] = G * dt

        # ∂q/∂q  — first-order integration of Omega matrix
        F[6:10, 6:10] = np.eye(4) + Omega * dt

        # ── Process noise Q  (10×10) ──────────────────────────────────────
        # Approximate: accel noise enters through velocity then position
        Qa = self._Q_accel
        Qg = self._Q_gyro
        Q = np.zeros((self.N, self.N))
        # velocity channel (via rotation)
        Q[3:6, 3:6]   = Qa * (R @ R.T) * dt**2   # ≈ Qa * I * dt²
        # position channel (second-order effect)
        Q[0:3, 0:3]   = Qa * (R @ R.T) * (dt**4 / 4)
        Q[0:3, 3:6]   = Qa * (R @ R.T) * (dt**3 / 2)
        Q[3:6, 0:3]   = Q[0:3, 3:6].T
        # quaternion channel
        Q[6:10, 6:10] = Qg * np.eye(4) * dt**2

        # ── covariance propagation ─────────────────────────────────────────
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Update step — radar measurement
    # ------------------------------------------------------------------

    def update(self, radar: RadarMeasurement) -> None:
        """Correct state using a spherical radar measurement."""
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]

        r_norm = np.linalg.norm(p)
        if r_norm < 1e-6:
            return

        px, py, pz = p
        r_xy = math.sqrt(px**2 + py**2)

        # ── predicted measurement  h(x) ───────────────────────────────────
        h_range     = r_norm
        h_azimuth   = math.atan2(py, px)
        h_elevation = math.asin(np.clip(pz / r_norm, -1.0, 1.0))
        unit_r      = p / r_norm
        h_doppler   = -float(np.dot(v, unit_r))

        z_hat = np.array([h_range, h_azimuth, h_elevation, h_doppler])

        # ── measurement Jacobian  H (4×10) ────────────────────────────────
        H = np.zeros((4, self.N))

        # ∂range/∂p
        H[0, 0:3] = p / r_norm

        # ∂azimuth/∂p
        r_xy2 = r_xy**2 if r_xy > 1e-8 else 1e-8
        H[1, 0]  = -py / r_xy2
        H[1, 1]  =  px / r_xy2

        # ∂elevation/∂p
        denom = r_norm**2 * r_xy if r_xy > 1e-8 else 1e-8
        H[2, 0]  = -px * pz / denom
        H[2, 1]  = -py * pz / denom
        H[2, 2]  =  r_xy / r_norm**2

        # ∂doppler/∂p  and  ∂doppler/∂v
        # doppler = -v·(p/|p|)  →  ∂/∂p  and  ∂/∂v
        H[3, 0:3] = (np.dot(v, p) * p / r_norm**3 - v / r_norm) * (-1)
        H[3, 3:6] = -unit_r

        # ── innovation ────────────────────────────────────────────────────
        z = np.array([
            radar.range,
            radar.azimuth,
            radar.elevation,
            radar.doppler,
        ])
        y = z - z_hat
        # wrap azimuth & elevation to (−π, π)
        y[1] = (y[1] + math.pi) % (2 * math.pi) - math.pi
        y[2] = (y[2] + math.pi) % (2 * math.pi) - math.pi

        # ── Kalman gain ───────────────────────────────────────────────────
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # ── state update ──────────────────────────────────────────────────
        self.x = self.x + K @ y
        self.x[6:10] = _quat_norm(self.x[6:10])

        # ── covariance update (Joseph form for numerical stability) ────────
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    # ------------------------------------------------------------------
    # State accessor
    # ------------------------------------------------------------------

    def get_state(self, timestamp: float) -> EkfState:
        """Package current state + covariance into an EkfState message."""
        return EkfState(
            timestamp=timestamp,
            pos_x=float(self.x[0]),
            pos_y=float(self.x[1]),
            pos_z=float(self.x[2]),
            vel_x=float(self.x[3]),
            vel_y=float(self.x[4]),
            vel_z=float(self.x[5]),
            qw=float(self.x[6]),
            qx=float(self.x[7]),
            qy=float(self.x[8]),
            qz=float(self.x[9]),
            covariance=self.P.flatten().tolist(),
        )
