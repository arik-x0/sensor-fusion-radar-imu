"""
IMU Service
===========
Simulates the rover's onboard IMU by subscribing to the rover_service ground
truth and converting it to noisy body-frame accelerometer + gyroscope
measurements.

The IMU is physically on the rover.  It measures:
  - Accelerometer: kinematic acceleration in body frame
      a_body = R^T · a_world
    where a_world is the rover's kinematic world-frame acceleration (gravity
    excluded, consistent with how the EKF predict step integrates it).
  - Gyroscope: body-frame angular rate (yaw/pitch/roll rates).

Rate limiting
-------------
The rover_service publishes at 200 Hz.  The IMU publishes at IMU_HZ (default
100 Hz) using a wall-clock gate, consuming but dropping every other truth
message.

Environment variables
---------------------
BROKER_XPUB_ADDR    ZMQ address to subscribe to (broker XPUB, default tcp://broker_service:5551)
BROKER_XSUB_ADDR    ZMQ address to publish to   (broker XSUB, default tcp://broker_service:5550)
IMU_HZ              Publication rate  (default 100)
IMU_NOISE_ACCEL     Accelerometer noise std-dev (m/s²)   (default 0.05)
IMU_NOISE_GYRO      Gyroscope noise std-dev (rad/s)      (default 0.005)
"""

from __future__ import annotations

import os
import time
import logging

import numpy as np
import zmq

from common.models import RoverTruth, ImuMeasurement
from common.topics import TOPIC_ROVER_TRUTH, TOPIC_IMU
from common.transport import Publisher, Subscriber

# ── configuration ─────────────────────────────────────────────────────────────
BROKER_XPUB_ADDR = os.getenv("BROKER_XPUB_ADDR", "tcp://broker_service:5551")
BROKER_XSUB_ADDR = os.getenv("BROKER_XSUB_ADDR", "tcp://broker_service:5550")
HZ               = float(os.getenv("IMU_HZ",          "100"))
NOISE_ACCEL      = float(os.getenv("IMU_NOISE_ACCEL", "0.05"))
NOISE_GYRO       = float(os.getenv("IMU_NOISE_GYRO",  "0.005"))

logging.basicConfig(
    level=logging.INFO,
    format="[imu_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3×3 rotation matrix.

    Returns R such that v_world = R @ v_body.
    Transpose (R.T) converts world → body.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)    ],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)    ],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def main() -> None:
    rng      = np.random.default_rng()
    min_dt   = 1.0 / HZ
    last_pub = 0.0

    log.info("Subscribing to rover truth on %s", BROKER_XPUB_ADDR)
    log.info("Publishing IMU measurements to %s at %.0f Hz", BROKER_XSUB_ADDR, HZ)

    sub = Subscriber(
        addresses=[BROKER_XPUB_ADDR],
        topics=[TOPIC_ROVER_TRUTH],
        recv_timeout_ms=2000,
    )

    log_counter = 0
    with Publisher(BROKER_XSUB_ADDR) as pub:
        while True:
            try:
                _, payload = sub.recv()
                truth = RoverTruth.from_bytes(payload)
            except zmq.Again:
                log.warning("No rover truth received within timeout — waiting…")
                continue
            except Exception as exc:
                log.error("Receive error: %s", exc)
                continue

            # Rate-gate: publish at IMU_HZ regardless of truth rate
            now = time.time()
            if now - last_pub < min_dt:
                continue
            last_pub = now

            # ── accelerometer: rotate world-frame accel into body frame ────
            a_world = np.array([
                truth.accel_world_x,
                truth.accel_world_y,
                truth.accel_world_z,
            ])
            q = np.array([truth.qw, truth.qx, truth.qy, truth.qz])
            R      = _quat_to_rot(q)    # body → world
            a_body = R.T @ a_world      # world → body

            noisy_accel = a_body + rng.normal(0.0, NOISE_ACCEL, 3)

            # ── gyroscope: body-frame angular rate ─────────────────────────
            gyro_true  = np.array([truth.gyro_x, truth.gyro_y, truth.gyro_z])
            noisy_gyro = gyro_true + rng.normal(0.0, NOISE_GYRO, 3)

            meas = ImuMeasurement(
                timestamp=truth.timestamp,
                accel_x=float(noisy_accel[0]),
                accel_y=float(noisy_accel[1]),
                accel_z=float(noisy_accel[2]),
                gyro_x=float(noisy_gyro[0]),
                gyro_y=float(noisy_gyro[1]),
                gyro_z=float(noisy_gyro[2]),
            )
            pub.send(TOPIC_IMU, meas.to_bytes())

            log_counter += 1
            if log_counter % 50 == 0:
                log.info(
                    "accel=(%.3f, %.3f, %.3f) m/s²  gyro=(%.4f, %.4f, %.4f) rad/s",
                    meas.accel_x, meas.accel_y, meas.accel_z,
                    meas.gyro_x,  meas.gyro_y,  meas.gyro_z,
                )


if __name__ == "__main__":
    main()
