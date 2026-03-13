"""
Radar Service
=============
Simulates a ground-based surveillance radar that tracks the rover published
by the rover_service.  The radar is fixed at the world-frame origin (0, 0, 0).

For each RoverTruth message received the service:
  1. Converts the rover's Cartesian position and velocity to spherical
     coordinates (range, azimuth, elevation, Doppler).
  2. Adds realistic Gaussian measurement noise.
  3. Rate-limits publication to RADAR_HZ (default 10 Hz) using a wall-clock
     gate — the rover_service publishes much faster (200 Hz), so most
     truth messages are consumed but silently dropped.

Radar model
-----------
  - Monostatic, ground-fixed at origin.
  - Measures: range (m), azimuth (rad), elevation (rad), Doppler (m/s).
  - Doppler sign convention: positive = target approaching.
  - The rover is a passive, cooperative target (no transponder needed for
    the radar measurement itself, but the rover_service does publish truth).

Environment variables
---------------------
BROKER_XPUB_ADDR     ZMQ address to subscribe to (broker XPUB, default tcp://broker_service:5551)
BROKER_XSUB_ADDR     ZMQ address to publish to   (broker XSUB, default tcp://broker_service:5550)
RADAR_HZ             Measurement publication rate (default 10)
RADAR_NOISE_RANGE    Range noise std-dev (m)       (default 1.5)
RADAR_NOISE_ANGLE    Angle noise std-dev (rad)     (default 0.01)
RADAR_NOISE_DOPPLER  Doppler noise std-dev (m/s)   (default 0.5)
"""

from __future__ import annotations

import math
import os
import time
import logging

import numpy as np
import zmq

from common.models import RoverTruth, RadarMeasurement
from common.topics import TOPIC_ROVER_TRUTH, TOPIC_RADAR
from common.transport import Publisher, Subscriber

# ── configuration ─────────────────────────────────────────────────────────────
BROKER_XPUB_ADDR = os.getenv("BROKER_XPUB_ADDR", "tcp://broker_service:5551")
BROKER_XSUB_ADDR = os.getenv("BROKER_XSUB_ADDR", "tcp://broker_service:5550")
HZ               = float(os.getenv("RADAR_HZ",            "10"))
NOISE_RANGE      = float(os.getenv("RADAR_NOISE_RANGE",   "1.5"))
NOISE_ANGLE      = float(os.getenv("RADAR_NOISE_ANGLE",   "0.01"))
NOISE_DOPPLER    = float(os.getenv("RADAR_NOISE_DOPPLER", "0.5"))

logging.basicConfig(
    level=logging.INFO,
    format="[radar_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _to_spherical(
    pos: np.ndarray, vel: np.ndarray
) -> tuple[float, float, float, float]:
    """Convert Cartesian position/velocity to (range, azimuth, elevation, doppler).

    The radar is at the origin.  Doppler is positive when the target
    approaches (radial velocity toward the radar).
    """
    r = float(np.linalg.norm(pos))
    if r < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    az      = math.atan2(pos[1], pos[0])
    el      = math.asin(float(np.clip(pos[2] / r, -1.0, 1.0)))
    unit_r  = pos / r
    doppler = -float(np.dot(vel, unit_r))   # positive = approaching
    return r, az, el, doppler


def main() -> None:
    rng      = np.random.default_rng()
    min_dt   = 1.0 / HZ
    last_pub = 0.0

    log.info("Subscribing to rover truth on %s", BROKER_XPUB_ADDR)
    log.info("Publishing radar measurements to %s at %.1f Hz", BROKER_XSUB_ADDR, HZ)

    sub = Subscriber(
        addresses=[BROKER_XPUB_ADDR],
        topics=[TOPIC_ROVER_TRUTH],
        recv_timeout_ms=2000,
    )

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

            # Rate-gate: only publish at RADAR_HZ regardless of truth rate
            now = time.time()
            if now - last_pub < min_dt:
                continue
            last_pub = now

            pos = np.array([truth.pos_x, truth.pos_y, truth.pos_z])
            vel = np.array([truth.vel_x, truth.vel_y, truth.vel_z])

            r, az, el, dp = _to_spherical(pos, vel)

            # Add measurement noise
            r  += rng.normal(0.0, NOISE_RANGE)
            az += rng.normal(0.0, NOISE_ANGLE)
            el += rng.normal(0.0, NOISE_ANGLE)
            dp += rng.normal(0.0, NOISE_DOPPLER)

            meas = RadarMeasurement(
                timestamp=truth.timestamp,
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


if __name__ == "__main__":
    main()
