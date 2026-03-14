"""
EKF2 Core — Error-State Extended Kalman Filter for Radar-IMU Sensor Fusion
===========================================================================

State vector  x ∈ ℝ¹⁶:
    [0:3]   position   p  (m)       — world/NED frame
    [3:6]   velocity   v  (m/s)     — world frame
    [6:10]  quaternion q  (w,x,y,z) — body-to-world orientation
    [10:13] accel bias b_a (m/s²)   — body-frame accelerometer bias
    [13:16] gyro  bias b_g (rad/s)  — body-frame gyroscope bias

Key EKF2 features over the baseline EKF
----------------------------------------
1. IMU bias estimation
       b_a and b_g are included in the state and estimated online.
       All IMU measurements are bias-corrected before integration,
       which prevents velocity and attitude drift accumulation.

2. Sequential scalar updates
       Each radar measurement component (range, azimuth, elevation,
       doppler) is processed independently as a 1-D observation.
       This improves numerical stability and allows per-component
       outlier gating without discarding the entire measurement.

3. Innovation gating
       Before applying each scalar update the normalised innovation²
       (y² / S) is compared against a chi-squared threshold.
       Measurements that exceed the gate are silently rejected,
       guarding against spurious radar returns.

References
----------
Sola, J. (2017). Quaternion kinematics for the error-state Kalman filter.
    arXiv:1711.02508  (Appendix E — quaternion Jacobians)
Groves, P.D. (2013). Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems. Chapter 14.
PX4 EKF2 implementation: https://github.com/PX4/PX4-ECL
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from common.models import ImuMeasurement, RadarMeasurement, EkfState

FloatArray = NDArray[np.float64]

# ── helpers ─────────────────────────────────────────────────────────────────

def _quat_norm(q: FloatArray) -> FloatArray:
    """Normalise quaternion; return identity if degenerate."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_to_rot(q: FloatArray) -> FloatArray:
    """Unit quaternion (w,x,y,z) → 3×3 rotation matrix.

    Rotates body-frame vectors into world frame: v_w = R @ v_b
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)    ],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)    ],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def _omega_matrix(w: FloatArray) -> FloatArray:
    """4×4 Omega matrix for quaternion kinematics: dq/dt = 0.5 Ω(ω) q."""
    gx, gy, gz = w
    return 0.5 * np.array([
        [ 0,  -gx, -gy, -gz],
        [ gx,  0,   gz, -gy],
        [ gy, -gz,  0,   gx],
        [ gz,  gy, -gx,  0  ],
    ])


def _xi_matrix(q: FloatArray) -> FloatArray:
    """4×3 matrix relating body angular-rate increments to quaternion increments.

    Derived from: d(Ω(ω)q)/dω  evaluated at quaternion q.
    Used for the Jacobian ∂q_new/∂b_g = −Xi(q)·dt.
    """
    w, x, y, z = q
    return 0.5 * np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w],
    ])


# ── EKF2 class ──────────────────────────────────────────────────────────────

class ExtendedKalmanFilter:
    """EKF2-style filter fusing IMU (predict) and radar (update) measurements.

    The state is augmented with IMU biases relative to the baseline EKF:

        x = [position(3), velocity(3), quaternion(4), accel_bias(3), gyro_bias(3)]

    Radar updates are applied sequentially (one scalar component at a time)
    with chi-squared innovation gating.

    Parameters
    ----------
    sigma_accel    : accelerometer measurement noise std-dev (m/s²)
    sigma_gyro     : gyroscope measurement noise std-dev (rad/s)
    sigma_range    : radar range noise std-dev (m)
    sigma_angle    : radar azimuth/elevation noise std-dev (rad)
    sigma_doppler  : radar Doppler noise std-dev (m/s)
    sigma_abias_rw : accel bias random-walk noise std-dev (m/s²/√s)
    sigma_gbias_rw : gyro bias random-walk noise std-dev (rad/s/√s)
    innov_gate     : innovation gate — reject scalar update when y²/S > gate
    """

    N = 16  # state dimension

    # ── state index slices ───────────────────────────────────────────────────
    _POS   = slice(0,  3)
    _VEL   = slice(3,  6)
    _QUAT  = slice(6,  10)
    _ABIAS = slice(10, 13)
    _GBIAS = slice(13, 16)

    def __init__(
        self,
        sigma_accel:    float = 0.1,
        sigma_gyro:     float = 0.01,
        sigma_range:    float = 2.0,
        sigma_angle:    float = 0.02,
        sigma_doppler:  float = 0.5,
        sigma_abias_rw: float = 1e-4,
        sigma_gbias_rw: float = 1e-5,
        innov_gate:     float = 5.0,
    ) -> None:
        # ── state ──────────────────────────────────────────────────────────
        self.x: FloatArray = np.zeros(self.N)
        self.x[0]  = 100.0  # initial position guess (m) — coarse
        self.x[6]  = 1.0    # quaternion w = 1 (identity rotation)
        # accel/gyro biases initialised to zero

        # ── covariance ─────────────────────────────────────────────────────
        self.P: FloatArray = np.diag([
            50**2, 50**2, 25**2,              # position (m²)
            10**2, 10**2,  5**2,              # velocity (m/s)²
            0.1**2, 0.1**2, 0.1**2, 0.1**2,  # quaternion
            0.5**2, 0.5**2, 0.5**2,           # accel bias (m/s²)²
            0.1**2, 0.1**2, 0.1**2,           # gyro bias (rad/s)²
        ]).astype(np.float64)

        # ── process noise (stored as variances) ────────────────────────────
        self._qa     = sigma_accel**2     # accelerometer noise variance
        self._qg     = sigma_gyro**2      # gyroscope noise variance
        self._qa_rw  = sigma_abias_rw**2  # accel bias random-walk PSD
        self._qg_rw  = sigma_gbias_rw**2  # gyro  bias random-walk PSD

        # ── measurement noise per radar component (scalar variances) ───────
        self._r_range   = sigma_range**2
        self._r_angle   = sigma_angle**2
        self._r_doppler = sigma_doppler**2

        # ── innovation gate ─────────────────────────────────────────────────
        # y²/S > _gate_sq → reject scalar measurement component
        self._gate_sq = innov_gate

        self._last_time: float | None = None

    # ------------------------------------------------------------------
    # Predict step — driven by IMU
    # ------------------------------------------------------------------

    def predict(self, imu: ImuMeasurement) -> None:
        """Propagate state forward using bias-corrected IMU measurements.

        Key difference from baseline EKF:
        - IMU measurements are corrected by the estimated bias (b_a, b_g)
          before integration, suppressing the drift that accumulates when
          raw measurements are integrated without bias compensation.
        - The state Jacobian F now includes bias coupling terms
          (∂v/∂b_a, ∂p/∂b_a, ∂q/∂b_g) so the covariance correctly
          accounts for the interaction between bias uncertainty and
          position/velocity/attitude uncertainty.
        """
        if self._last_time is None:
            self._last_time = imu.timestamp
            return

        dt = imu.timestamp - self._last_time
        if dt <= 0.0 or dt > 1.0:          # ignore stale / future packets
            self._last_time = imu.timestamp
            return
        self._last_time = imu.timestamp

        p   = self.x[self._POS]
        v   = self.x[self._VEL]
        q   = self.x[self._QUAT]
        b_a = self.x[self._ABIAS]
        b_g = self.x[self._GBIAS]

        # ── bias-corrected IMU readings ───────────────────────────────────
        a_body = np.array([imu.accel_x, imu.accel_y, imu.accel_z]) - b_a
        w_body = np.array([imu.gyro_x,  imu.gyro_y,  imu.gyro_z ]) - b_g

        R       = _quat_to_rot(q)
        a_world = R @ a_body          # rotate corrected accel into world frame

        # ── state propagation (Euler integration) ─────────────────────────
        p_new = p + v * dt + 0.5 * a_world * dt**2
        v_new = v + a_world * dt
        Omega = _omega_matrix(w_body)
        q_new = _quat_norm(q + (Omega @ q) * dt)
        # biases: random-walk model — nominal value unchanged, only P grows

        self.x[self._POS]  = p_new
        self.x[self._VEL]  = v_new
        self.x[self._QUAT] = q_new
        # self.x[_ABIAS] and self.x[_GBIAS] unchanged

        # ── Jacobian F of f(x) w.r.t. x  (16×16) ─────────────────────────
        F = np.eye(self.N)

        # ∂p/∂v
        F[0:3, 3:6] = np.eye(3) * dt

        # ∂v/∂q  and  ∂p/∂q  (closed-form from Sola 2017, Appendix E)
        # G is the 3×4 Jacobian: d(R(q) @ a_body) / dq
        w_, x_, y_, z_ = q
        a = a_body  # use bias-corrected accel as linearisation point
        G = 2.0 * np.column_stack([
            # ∂/∂qw                  # ∂/∂qx                              # ∂/∂qy                            # ∂/∂qz
            np.cross(np.array([x_, y_, z_]), a),
            np.array([ x_*a[0]+y_*a[1]+z_*a[2],  y_*a[0]-x_*a[1]-w_*a[2],  z_*a[0]+w_*a[1]-x_*a[2]]),
            np.array([-y_*a[0]+x_*a[1]+w_*a[2],  x_*a[0]+y_*a[1]+z_*a[2], -w_*a[0]+z_*a[1]-y_*a[2]]),
            np.array([-z_*a[0]-w_*a[1]+x_*a[2],  w_*a[0]-z_*a[1]+y_*a[2],  x_*a[0]+y_*a[1]+z_*a[2]]),
        ])
        F[3:6, 6:10] = G * dt
        F[0:3, 6:10] = G * (dt**2 / 2.0)

        # ∂q/∂q  — first-order integration of Omega matrix
        F[6:10, 6:10] = np.eye(4) + Omega * dt

        # ── bias coupling terms (unique to EKF2) ──────────────────────────
        # ∂v/∂b_a  and  ∂p/∂b_a
        # a_world = R @ (a_body_raw - b_a)  →  ∂a_world/∂b_a = -R
        F[3:6, 10:13] = -R * dt
        F[0:3, 10:13] = -R * (dt**2 / 2.0)

        # ∂q/∂b_g
        # q_new ≈ q + 0.5·Ω(w_body - b_g)·q·dt  →  ∂q_new/∂b_g = -Xi(q)·dt
        Xi = _xi_matrix(q)
        F[6:10, 13:16] = -Xi * dt

        # ── Process noise Q  (16×16) ──────────────────────────────────────
        Q = np.zeros((self.N, self.N))

        # Accelerometer noise → velocity and position channels
        Q[3:6, 3:6]  = self._qa * np.eye(3) * dt**2
        Q[0:3, 0:3]  = self._qa * np.eye(3) * (dt**4 / 4.0)
        Q[0:3, 3:6]  = self._qa * np.eye(3) * (dt**3 / 2.0)
        Q[3:6, 0:3]  = Q[0:3, 3:6].T

        # Gyroscope noise → quaternion channel
        Q[6:10, 6:10] = self._qg * np.eye(4) * dt**2

        # Bias random walks — variance grows as σ²·dt per step
        Q[10:13, 10:13] = self._qa_rw * np.eye(3) * dt
        Q[13:16, 13:16] = self._qg_rw * np.eye(3) * dt

        # ── covariance propagation ─────────────────────────────────────────
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Internal: predict measurement and Jacobian from current state
    # ------------------------------------------------------------------

    def _predict_measurement(self) -> tuple[FloatArray, FloatArray] | None:
        """Compute predicted radar measurement h(x) and Jacobian H (4×16).

        Returns None when the target is too close to the origin to compute
        spherical coordinates reliably.
        """
        p = self.x[self._POS]
        v = self.x[self._VEL]

        r_norm = np.linalg.norm(p)
        if r_norm < 1e-6:
            return None

        px, py, pz = p
        r_xy   = math.sqrt(px**2 + py**2)
        unit_r = p / r_norm

        # Predicted measurement vector
        h = np.array([
            r_norm,
            math.atan2(py, px),
            math.asin(np.clip(pz / r_norm, -1.0, 1.0)),
            -float(np.dot(v, unit_r)),
        ])

        # Measurement Jacobian H (4×16) — only position and velocity columns
        H = np.zeros((4, self.N))

        # ∂range/∂p
        H[0, 0:3] = p / r_norm

        # ∂azimuth/∂p
        r_xy2    = r_xy**2 if r_xy > 1e-8 else 1e-8
        H[1, 0]  = -py / r_xy2
        H[1, 1]  =  px / r_xy2

        # ∂elevation/∂p
        denom    = r_norm**2 * r_xy if r_xy > 1e-8 else 1e-8
        H[2, 0]  = -px * pz / denom
        H[2, 1]  = -py * pz / denom
        H[2, 2]  =  r_xy / r_norm**2

        # ∂doppler/∂p  and  ∂doppler/∂v
        H[3, 0:3] = (np.dot(v, p) * p / r_norm**3 - v / r_norm) * (-1)
        H[3, 3:6] = -unit_r

        return h, H

    # ------------------------------------------------------------------
    # Update step — radar measurement (sequential scalar updates)
    # ------------------------------------------------------------------

    def update(self, radar: RadarMeasurement) -> None:
        """Correct state using spherical radar measurements.

        Key differences from baseline EKF:

        Sequential scalar updates
            Instead of a single 4×16 Kalman gain, each of the four
            radar components (range, azimuth, elevation, doppler) is
            applied as an independent 1-D observation.  After each
            scalar update the predicted measurement is recomputed from
            the updated state so that the next component uses the most
            current linearisation point.  This mirrors the approach
            used in PX4's EKF2 and is more numerically stable than
            the batch update.

        Innovation gating
            Each scalar innovation y_i is tested against a chi-squared
            gate:  y_i² / S_i > gate_threshold → component rejected.
            Radar outliers (clutter, multipath) typically produce large
            innovations in one component; sequential gating lets the
            remaining components still contribute.
        """
        z = np.array([
            radar.range,
            radar.azimuth,
            radar.elevation,
            radar.doppler,
        ])
        r_per_component = [
            self._r_range,
            self._r_angle,
            self._r_angle,
            self._r_doppler,
        ]
        wrap_angle = [False, True, True, False]

        for i in range(4):
            result = self._predict_measurement()
            if result is None:
                return
            h, H = result
            H_i = H[i]           # 1×16 row vector as a 1-D array

            y_i = z[i] - h[i]
            if wrap_angle[i]:
                y_i = (y_i + math.pi) % (2 * math.pi) - math.pi

            # ── innovation covariance (scalar) ────────────────────────────
            PHt = self.P @ H_i          # (16,) vector
            S_i = float(H_i @ PHt) + r_per_component[i]
            if S_i < 1e-10:
                continue

            # ── innovation gate (chi-squared, 1 DOF) ─────────────────────
            if (y_i**2 / S_i) > self._gate_sq:
                continue

            # ── Kalman gain (16,) vector ───────────────────────────────────
            K_i = PHt / S_i

            # ── state update ──────────────────────────────────────────────
            self.x = self.x + K_i * y_i
            self.x[self._QUAT] = _quat_norm(self.x[self._QUAT])

            # ── covariance update (Joseph form for numerical stability) ────
            I_KH = np.eye(self.N) - np.outer(K_i, H_i)
            self.P = I_KH @ self.P @ I_KH.T + r_per_component[i] * np.outer(K_i, K_i)

    # ------------------------------------------------------------------
    # State accessor
    # ------------------------------------------------------------------

    def get_state(self, timestamp: float) -> EkfState:
        """Package current state + covariance into an EkfState message.

        The covariance field is the 10×10 position/velocity/attitude block
        of the full 16×16 covariance matrix (first 10 states), preserving
        compatibility with the monitor and visualizer services.
        """
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
            covariance=self.P[:10, :10].flatten().tolist(),
        )
