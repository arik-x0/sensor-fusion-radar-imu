"""
Control Motion Rover Service
==============================
Publishes the motion Rover's true kinematic state at a high rate so that
both radar_service and imu_service derive their measurements from the same
physical Rover.

The motion Rover flies a smooth circular orbit with a slow altitude
oscillation — a trajectory that is fully analytical (no numerical integration)
so both sensor services always see a consistent, ground-truth state regardless
of when they query it.

Trajectory (world frame, origin = radar ground station)
--------------------------------------------------------
  x(t) = Cx + R · cos(ω·t + φ)
  y(t) = Cy + R · sin(ω·t + φ)
  z(t) = h₀ + h_amp · sin(ωz·t)

Orientation
-----------
  Yaw  = direction of horizontal velocity (nose points where the Rover moves).
  Pitch ≈ 0  (level flight approximation; small vertical velocity ignored).
  Roll  ≈ 0  (banking neglected for the simulation model).

Angular rate (body frame)
-------------------------
  gyro_z = ω  (constant yaw rate for circular orbit)
  gyro_y = pitch rate derived from the altitude oscillation
  gyro_x = 0  (no roll rate)

Environment variables
---------------------
BROKER_XSUB_ADDR   ZMQ connect address for broker XSUB  (default tcp://broker_service:5550)
ROVER_HZ          Truth publication rate (Hz)           (default 200)
ORBIT_RADIUS       Orbit radius in metres                (default 150)
ORBIT_OMEGA        Orbit angular velocity in rad/s       (default 0.15)
ORBIT_CX           Orbit centre x in metres              (default 0)
ORBIT_CY           Orbit centre y in metres              (default 0)
ROVER_ALTITUDE    Base altitude in metres               (default 80)
ROVER_ALT_AMP     Altitude oscillation amplitude (m)    (default 20)
ROVER_ALT_OMEGA   Altitude oscillation frequency (rad/s)(default 0.05)
"""

from __future__ import annotations

import math
import os
import time
import logging

import numpy as np

from common.models import RoverTruth
from common.topics import TOPIC_ROVER_TRUTH
from common.transport import Publisher

# ── configuration ─────────────────────────────────────────────────────────────
BROKER_XSUB_ADDR  = os.getenv("BROKER_XSUB_ADDR",  "tcp://broker_service:5550")
ROVER_HZ         = float(os.getenv("ROVER_HZ",        "200"))
ORBIT_RADIUS      = float(os.getenv("ORBIT_RADIUS",     "150.0"))
ORBIT_OMEGA       = float(os.getenv("ORBIT_OMEGA",      "0.15"))
ORBIT_PHI         = float(os.getenv("ORBIT_PHI",        "0.0"))
ORBIT_CX          = float(os.getenv("ORBIT_CX",         "0.0"))
ORBIT_CY          = float(os.getenv("ORBIT_CY",         "0.0"))
ROVER_ALTITUDE   = float(os.getenv("ROVER_ALTITUDE",  "80.0"))
ROVER_ALT_AMP    = float(os.getenv("ROVER_ALT_AMP",   "20.0"))
ROVER_ALT_OMEGA  = float(os.getenv("ROVER_ALT_OMEGA", "0.05"))

logging.basicConfig(
    level=logging.INFO,
    format="[rover_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── quaternion helpers ────────────────────────────────────────────────────────

def _quat_from_yaw(yaw: float) -> np.ndarray:
    """Build a unit quaternion (w, x, y, z) for a pure yaw rotation."""
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


# ── trajectory ────────────────────────────────────────────────────────────────

class RoverTrajectory:
    """Analytic circular orbit with altitude oscillation.

    All quantities are derived analytically from the current time, so any
    service can independently compute the motion ROVER's state at any moment
    and get a consistent result.
    """

    def __init__(self, t0: float) -> None:
        self._t0 = t0  # wall-clock time when the simulation started

    def state_at(self, t_abs: float) -> RoverTruth:
        """Return the motion ROVER's true state at absolute wall-clock time *t_abs*."""
        t = t_abs - self._t0
        theta = ORBIT_OMEGA * t + ORBIT_PHI

        # ── position (world frame, m) ─────────────────────────────────────
        px = ORBIT_CX + ORBIT_RADIUS * math.cos(theta)
        py = ORBIT_CY + ORBIT_RADIUS * math.sin(theta)
        pz = ROVER_ALTITUDE + ROVER_ALT_AMP * math.sin(ROVER_ALT_OMEGA * t)

        # ── velocity (world frame, m/s) ───────────────────────────────────
        vx = -ORBIT_RADIUS * ORBIT_OMEGA * math.sin(theta)
        vy =  ORBIT_RADIUS * ORBIT_OMEGA * math.cos(theta)
        vz =  ROVER_ALT_AMP * ROVER_ALT_OMEGA * math.cos(ROVER_ALT_OMEGA * t)

        # ── kinematic acceleration (world frame, m/s²) — gravity excluded ─
        ax = -ORBIT_RADIUS * ORBIT_OMEGA ** 2 * math.cos(theta)
        ay = -ORBIT_RADIUS * ORBIT_OMEGA ** 2 * math.sin(theta)
        az = -ROVER_ALT_AMP * ROVER_ALT_OMEGA ** 2 * math.sin(ROVER_ALT_OMEGA * t)

        # ── orientation: yaw follows velocity direction ───────────────────
        yaw = math.atan2(vy, vx)
        q = _quat_from_yaw(yaw)

        # ── angular rate (body frame, rad/s) ─────────────────────────────
        # Yaw rate = orbital angular velocity (constant for circular orbit).
        # Pitch rate = rate of change of the pitch angle (≈ arctan(vz / v_xy)).
        v_xy = math.hypot(vx, vy)
        if v_xy > 0.1:
            # d/dt [atan2(vz, v_xy)] via chain rule
            pitch_rate = (az * v_xy - vz * (ax * vx / v_xy + ay * vy / v_xy)) / (
                v_xy ** 2 + vz ** 2
            )
        else:
            pitch_rate = 0.0

        return RoverTruth(
            timestamp=t_abs,
            pos_x=px, pos_y=py, pos_z=pz,
            vel_x=vx, vel_y=vy, vel_z=vz,
            accel_world_x=ax, accel_world_y=ay, accel_world_z=az,
            qw=float(q[0]), qx=float(q[1]), qy=float(q[2]), qz=float(q[3]),
            gyro_x=0.0,          # no roll rate (level flight approx.)
            gyro_y=pitch_rate,
            gyro_z=ORBIT_OMEGA,  # constant yaw rate = orbital angular velocity
        )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info(
        "Motion ROVER simulator starting — orbit R=%.0f m  ω=%.3f rad/s  alt=%.0f±%.0f m",
        ORBIT_RADIUS, ORBIT_OMEGA, ROVER_ALTITUDE, ROVER_ALT_AMP,
    )
    log.info("Publishing to broker %s at %.0f Hz on topic '%s'",
             BROKER_XSUB_ADDR, ROVER_HZ, TOPIC_ROVER_TRUTH)

    dt         = 1.0 / ROVER_HZ
    t0         = time.time()
    trajectory = RoverTrajectory(t0)

    with Publisher(BROKER_XSUB_ADDR) as pub:
        log_counter = 0
        while True:
            loop_start = time.monotonic()

            truth = trajectory.state_at(time.time())
            pub.send(TOPIC_ROVER_TRUTH, truth.to_bytes())

            log_counter += 1
            if log_counter % int(ROVER_HZ) == 0:  # once per second
                log.info(
                    "pos=(%.1f, %.1f, %.1f) m  vel=(%.2f, %.2f, %.2f) m/s",
                    truth.pos_x, truth.pos_y, truth.pos_z,
                    truth.vel_x, truth.vel_y, truth.vel_z,
                )

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
