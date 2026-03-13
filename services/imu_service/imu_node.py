"""
IMU Simulator Service
=====================
Simulates an IMU mounted on a body undergoing slow constant acceleration
with a gentle rotation, and publishes noisy accel + gyro measurements.

Publishes to the broker XSUB port so that all subscribers (buffer service,
monitors) receive measurements through the central broker.

Environment variables
---------------------
BROKER_XSUB_ADDR    ZMQ connect address for broker XSUB  (default tcp://broker_service:5550)
IMU_HZ              Publication rate  (default 100)
IMU_NOISE_ACCEL     Std-dev accel noise  m/s²  (default 0.05)
IMU_NOISE_GYRO      Std-dev gyro  noise  rad/s (default 0.005)
"""

import os
import time
import numpy as np
import logging

from common.models import ImuMeasurement
from common.transport import Publisher
from common.topics import TOPIC_IMU

# ── configuration ──────────────────────────────────────────────────────────
BROKER_XSUB_ADDR = os.getenv("BROKER_XSUB_ADDR",  "tcp://broker_service:5550")
HZ               = float(os.getenv("IMU_HZ",          "100"))
NOISE_ACCEL      = float(os.getenv("IMU_NOISE_ACCEL", "0.05"))
NOISE_GYRO       = float(os.getenv("IMU_NOISE_GYRO",  "0.005"))

logging.basicConfig(
    level=logging.INFO,
    format="[imu_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── simulated body dynamics ─────────────────────────────────────────────────
# True acceleration and angular rate of the simulated body.
# These represent what an ideal IMU would measure (before gravity / noise).
TRUE_ACCEL = np.array([0.3, 0.1, -0.05])   # m/s² (body frame)
TRUE_GYRO  = np.array([0.02, -0.01, 0.005]) # rad/s (slow yaw drift)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng()
    dt  = 1.0 / HZ

    log.info("Connecting publisher to broker %s at %.1f Hz", BROKER_XSUB_ADDR, HZ)

    with Publisher(BROKER_XSUB_ADDR) as pub:
        while True:
            t0 = time.monotonic()

            noisy_accel = TRUE_ACCEL + rng.normal(0.0, NOISE_ACCEL, 3)
            noisy_gyro  = TRUE_GYRO  + rng.normal(0.0, NOISE_GYRO,  3)

            meas = ImuMeasurement(
                timestamp=time.time(),
                accel_x=float(noisy_accel[0]),
                accel_y=float(noisy_accel[1]),
                accel_z=float(noisy_accel[2]),
                gyro_x=float(noisy_gyro[0]),
                gyro_y=float(noisy_gyro[1]),
                gyro_z=float(noisy_gyro[2]),
            )
            pub.send(TOPIC_IMU, meas.to_bytes())

            # only log every 50 packets to avoid terminal flood
            if int(time.monotonic() * HZ) % 50 == 0:
                log.info(
                    "accel=(%.3f, %.3f, %.3f) gyro=(%.4f, %.4f, %.4f)",
                    meas.accel_x, meas.accel_y, meas.accel_z,
                    meas.gyro_x,  meas.gyro_y,  meas.gyro_z,
                )

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
