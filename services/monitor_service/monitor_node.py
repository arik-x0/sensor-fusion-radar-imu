"""
State Monitor Service
=====================
Subscribes to the EKF state topic and pretty-prints the fused estimate
to stdout, including a covariance trace for a quick health check.

Environment variables
---------------------
BROKER_XPUB_ADDR    ZMQ connect address for broker XPUB  (default tcp://broker_service:5551)
"""

import os
import math
import time
import logging
import zmq

from common.models import EkfState
from common.transport import Subscriber
from common.topics import TOPIC_STATE

BROKER_XPUB_ADDR = os.getenv("BROKER_XPUB_ADDR", "tcp://broker_service:5551")

logging.basicConfig(
    level=logging.INFO,
    format="[monitor_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ANSI colour helpers (safe on most terminals)
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"


def _cov_trace(state: EkfState) -> float:
    import numpy as np
    P = np.array(state.covariance).reshape(10, 10)
    return float(np.trace(P))


def _quat_to_euler_deg(qw, qx, qy, qz):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
    sin_r_cos_p = 2 * (qw * qx + qy * qz)
    cos_r_cos_p = 1 - 2 * (qx * qx + qy * qy)
    roll  = math.degrees(math.atan2(sin_r_cos_p, cos_r_cos_p))

    sin_p = 2 * (qw * qy - qz * qx)
    sin_p = max(-1.0, min(1.0, sin_p))
    pitch = math.degrees(math.asin(sin_p))

    sin_y_cos_p = 2 * (qw * qz + qx * qy)
    cos_y_cos_p = 1 - 2 * (qy * qy + qz * qz)
    yaw   = math.degrees(math.atan2(sin_y_cos_p, cos_y_cos_p))

    return roll, pitch, yaw


def main():
    log.info("Connecting to broker at %s", BROKER_XPUB_ADDR)

    sub = Subscriber(
        addresses=[BROKER_XPUB_ADDR],
        topics=[TOPIC_STATE],
        recv_timeout_ms=5000,
    )

    count = 0
    while True:
        try:
            _, payload = sub.recv()
            state = EkfState.from_bytes(payload)
            count += 1

            roll, pitch, yaw = _quat_to_euler_deg(
                state.qw, state.qx, state.qy, state.qz
            )
            trace = _cov_trace(state)

            # ── pretty print ──────────────────────────────────────────────
            print(
                f"\n{_CYAN}━━━ EKF State #{count:>5}  "
                f"t={state.timestamp:.3f} ━━━{_RESET}\n"
                f"  {_GREEN}Position  {_RESET}"
                f"x={state.pos_x:>9.3f} m   "
                f"y={state.pos_y:>9.3f} m   "
                f"z={state.pos_z:>9.3f} m\n"
                f"  {_GREEN}Velocity  {_RESET}"
                f"x={state.vel_x:>9.3f} m/s "
                f"y={state.vel_y:>9.3f} m/s "
                f"z={state.vel_z:>9.3f} m/s\n"
                f"  {_GREEN}Attitude  {_RESET}"
                f"roll={roll:>7.2f}°  pitch={pitch:>7.2f}°  yaw={yaw:>7.2f}°\n"
                f"  {_YELLOW}Cov trace {trace:.4e}{_RESET}",
                flush=True,
            )

        except zmq.Again:
            log.warning("No state received from EKF service within 5 s — waiting…")
        except Exception as exc:
            log.error("Monitor error: %s", exc)


if __name__ == "__main__":
    main()
