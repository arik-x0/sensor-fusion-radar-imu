"""
Radar Simulator Service
=======================
Simulates a target moving through 3D space and publishes noisy radar
measurements (range, azimuth, elevation, Doppler) at a configurable rate.

Publishes to the broker XSUB port so that all subscribers (buffer service,
monitors) receive measurements through the central broker.

Environment variables
---------------------
BROKER_XSUB_ADDR     ZMQ connect address for broker XSUB  (default tcp://broker_service:5550)
RADAR_HZ             Publication rate  (default 10)
RADAR_NOISE_RANGE    Std-dev of range noise in metres   (default 1.5)
RADAR_NOISE_ANGLE    Std-dev of angle noise in radians  (default 0.01)
RADAR_NOISE_DOPPLER  Std-dev of Doppler noise in m/s    (default 0.5)
"""

import os
import math
import time
import numpy as np
import logging

from common.models import RadarMeasurement
from common.transport import Publisher
from common.topics import TOPIC_RADAR

# ── configuration ──────────────────────────────────────────────────────────
BROKER_XSUB_ADDR = os.getenv("BROKER_XSUB_ADDR",    "tcp://broker_service:5550")
HZ               = float(os.getenv("RADAR_HZ",           "10"))
NOISE_RANGE      = float(os.getenv("RADAR_NOISE_RANGE",  "1.5"))
NOISE_ANGLE      = float(os.getenv("RADAR_NOISE_ANGLE",  "0.01"))
NOISE_DOPPLER    = float(os.getenv("RADAR_NOISE_DOPPLER","0.5"))

logging.basicConfig(
    level=logging.INFO,
    format="[radar_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── simulated target dynamics ───────────────────────────────────────────────
class TargetSimulator:
    """Simple constant-velocity target in 3-D space."""

    def __init__(self):
        # initial position (m) and velocity (m/s)
        self.pos = np.array([100.0, 50.0, 20.0])
        self.vel = np.array([-3.0,  1.5, -0.5])

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance simulation by *dt* seconds; return (position, velocity)."""
        self.pos = self.pos + self.vel * dt
        return self.pos.copy(), self.vel.copy()


def cartesian_to_spherical(pos: np.ndarray, vel: np.ndarray):
    """Convert Cartesian position/velocity to (range, azimuth, elevation, doppler)."""
    r = np.linalg.norm(pos)
    if r < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    az  = math.atan2(pos[1], pos[0])                    # radians
    el  = math.asin(np.clip(pos[2] / r, -1.0, 1.0))    # radians
    # Radial velocity (positive = approaching → negate the dot product)
    unit_r  = pos / r
    doppler = -float(np.dot(vel, unit_r))               # m/s
    return float(r), float(az), float(el), float(doppler)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    rng    = np.random.default_rng()
    target = TargetSimulator()
    dt     = 1.0 / HZ

    log.info("Connecting publisher to broker %s at %.1f Hz", BROKER_XSUB_ADDR, HZ)

    with Publisher(BROKER_XSUB_ADDR) as pub:
        while True:
            t0 = time.monotonic()

            pos, vel = target.step(dt)
            r, az, el, dp = cartesian_to_spherical(pos, vel)

            # add measurement noise
            r  += rng.normal(0.0, NOISE_RANGE)
            az += rng.normal(0.0, NOISE_ANGLE)
            el += rng.normal(0.0, NOISE_ANGLE)
            dp += rng.normal(0.0, NOISE_DOPPLER)

            meas = RadarMeasurement(
                timestamp=time.time(),
                range=max(0.0, r),
                azimuth=az,
                elevation=el,
                doppler=dp,
            )
            pub.send(TOPIC_RADAR, meas.to_bytes())
            log.info(
                "r=%.2f m  az=%.3f rad  el=%.3f rad  dp=%.2f m/s",
                meas.range, meas.azimuth, meas.elevation, meas.doppler,
            )

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
