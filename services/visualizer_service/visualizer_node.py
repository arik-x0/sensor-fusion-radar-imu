"""
Visualizer Service
==================
Subscribes to both TOPIC_ROVER_TRUTH (ground truth) and TOPIC_STATE (EKF
estimate) and renders a live 3-D matplotlib plot overlaying both trajectories
so you can directly compare how closely the EKF tracks the real rover.

  Green  = rover_service ground truth  (where the rover actually is)
  Blue   = EKF estimate               (what the system thinks)
  Orange = current velocity vector of the EKF estimate

The info box shows the real-time position error (Euclidean distance between
truth and estimate) so you can watch the filter converge.

Designed to run locally — requires a graphical display.
Not suitable for headless Docker without X11 forwarding / Xvfb.

Environment variables
---------------------
BROKER_XPUB_ADDR   ZMQ connect address for broker XPUB  (default tcp://localhost:5551)
VIZ_HISTORY        Number of past positions kept in each trail  (default 500)
VIZ_UPDATE_MS      Plot refresh interval in milliseconds        (default 100)
"""

from __future__ import annotations

import collections
import math
import os
import threading
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import zmq
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3-D projection

from common.models import RoverTruth, EkfState
from common.topics import TOPIC_ROVER_TRUTH, TOPIC_STATE
from common.transport import Subscriber

BROKER_XPUB_ADDR = os.getenv("BROKER_XPUB_ADDR", "tcp://localhost:5551")
VIZ_HISTORY      = int(os.getenv("VIZ_HISTORY",    "500"))
VIZ_UPDATE_MS    = int(os.getenv("VIZ_UPDATE_MS",  "100"))


# ── Thread-safe ring buffers ──────────────────────────────────────────────────

@dataclass
class _Point3:
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    cov_trace: float = 0.0


class _TrailBuffer:
    def __init__(self, maxlen: int) -> None:
        self._lock   = threading.Lock()
        self._trail: collections.deque[_Point3] = collections.deque(maxlen=maxlen)

    def push(self, pt: _Point3) -> None:
        with self._lock:
            self._trail.append(pt)

    def snapshot(self) -> list[_Point3]:
        with self._lock:
            return list(self._trail)


# ── Subscriber thread ─────────────────────────────────────────────────────────

def _subscriber_thread(truth_buf: _TrailBuffer, ekf_buf: _TrailBuffer) -> None:
    sub = Subscriber(
        addresses=[BROKER_XPUB_ADDR],
        topics=[TOPIC_ROVER_TRUTH, TOPIC_STATE],
        recv_timeout_ms=2000,
    )
    while True:
        try:
            topic, payload = sub.recv()
            if topic == TOPIC_ROVER_TRUTH:
                t = RoverTruth.from_bytes(payload)
                truth_buf.push(_Point3(
                    x=t.pos_x, y=t.pos_y, z=t.pos_z,
                    vx=t.vel_x, vy=t.vel_y, vz=t.vel_z,
                ))
            elif topic == TOPIC_STATE:
                s = EkfState.from_bytes(payload)
                cov = float(np.trace(np.array(s.covariance).reshape(10, 10)))
                ekf_buf.push(_Point3(
                    x=s.pos_x, y=s.pos_y, z=s.pos_z,
                    vx=s.vel_x, vy=s.vel_y, vz=s.vel_z,
                    cov_trace=cov,
                ))
        except zmq.Again:
            pass
        except Exception:
            pass


# ── Plot ──────────────────────────────────────────────────────────────────────

def main() -> None:
    truth_buf = _TrailBuffer(maxlen=VIZ_HISTORY)
    ekf_buf   = _TrailBuffer(maxlen=VIZ_HISTORY)

    threading.Thread(
        target=_subscriber_thread,
        args=(truth_buf, ekf_buf),
        daemon=True,
        name="viz-sub",
    ).start()

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        "Drone Tracking — Ground Truth vs EKF Estimate",
        fontsize=14, fontweight="bold",
    )

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # ── Ground truth artists (green) ─────────────────────────────────────────
    (truth_trail,) = ax.plot([], [], [], "-",
                             color="limegreen", lw=1.5, alpha=0.7,
                             label="True trajectory")
    (truth_dot,)   = ax.plot([], [], [], "^",
                             color="darkgreen", ms=9,
                             label="True position")

    # ── EKF estimate artists (blue) ───────────────────────────────────────────
    (ekf_trail,)   = ax.plot([], [], [], "-",
                             color="steelblue", lw=1.5, alpha=0.7,
                             label="EKF estimate")
    (ekf_dot,)     = ax.plot([], [], [], "o",
                             color="crimson", ms=9,
                             label="EKF position")

    # EKF velocity arrow — replaced each frame
    ekf_quiver: list = [None]

    # Error line connecting truth to estimate at current moment
    (error_line,)  = ax.plot([], [], [], "--",
                             color="gold", lw=1.5, alpha=0.9,
                             label="Position error")

    # Info overlay
    info_text = ax.text2D(
        0.02, 0.97, "",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                  edgecolor="gray", alpha=0.75),
        color="white",
    )

    ax.legend(loc="upper right", fontsize=8)

    def _update(_frame):
        truths = truth_buf.snapshot()
        ekfs   = ekf_buf.snapshot()

        if not truths and not ekfs:
            return truth_trail, truth_dot, ekf_trail, ekf_dot, error_line, info_text

        # ── Truth trail ───────────────────────────────────────────────────────
        if truths:
            txs = np.array([p.x for p in truths])
            tys = np.array([p.y for p in truths])
            tzs = np.array([p.z for p in truths])
            truth_trail.set_data(txs, tys)
            truth_trail.set_3d_properties(tzs)
            truth_dot.set_data([txs[-1]], [tys[-1]])
            truth_dot.set_3d_properties([tzs[-1]])

        # ── EKF trail ─────────────────────────────────────────────────────────
        if ekfs:
            exs = np.array([p.x for p in ekfs])
            eys = np.array([p.y for p in ekfs])
            ezs = np.array([p.z for p in ekfs])
            ekf_trail.set_data(exs, eys)
            ekf_trail.set_3d_properties(ezs)
            ekf_dot.set_data([exs[-1]], [eys[-1]])
            ekf_dot.set_3d_properties([ezs[-1]])

            # Velocity arrow
            if ekf_quiver[0] is not None:
                ekf_quiver[0].remove()
            last_ekf = ekfs[-1]
            ekf_quiver[0] = ax.quiver(
                last_ekf.x, last_ekf.y, last_ekf.z,
                last_ekf.vx, last_ekf.vy, last_ekf.vz,
                color="darkorange", linewidth=2, arrow_length_ratio=0.3,
            )

        # ── Error line between current truth and current estimate ─────────────
        pos_error_m = float("nan")
        if truths and ekfs:
            t_last = truths[-1]
            e_last = ekfs[-1]
            error_line.set_data([t_last.x, e_last.x], [t_last.y, e_last.y])
            error_line.set_3d_properties([t_last.z, e_last.z])
            pos_error_m = math.sqrt(
                (t_last.x - e_last.x) ** 2 +
                (t_last.y - e_last.y) ** 2 +
                (t_last.z - e_last.z) ** 2
            )

        # ── Axis limits — fit both trails ─────────────────────────────────────
        all_x = (
            [p.x for p in truths] + [p.x for p in ekfs]
        ) or [0.0]
        all_y = (
            [p.y for p in truths] + [p.y for p in ekfs]
        ) or [0.0]
        all_z = (
            [p.z for p in truths] + [p.z for p in ekfs]
        ) or [0.0]
        pad = max(5.0, (max(all_x) - min(all_x) +
                        max(all_y) - min(all_y) +
                        max(all_z) - min(all_z)) * 0.08)
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_zlim(min(all_z) - pad, max(all_z) + pad)

        # ── Info box ──────────────────────────────────────────────────────────
        cov_str = f"{ekfs[-1].cov_trace:.3e}" if ekfs else "—"
        err_str = f"{pos_error_m:.2f} m" if not math.isnan(pos_error_m) else "—"
        t_pos = f"({truths[-1].x:+.1f}, {truths[-1].y:+.1f}, {truths[-1].z:+.1f})" if truths else "—"
        e_pos = f"({ekfs[-1].x:+.1f}, {ekfs[-1].y:+.1f}, {ekfs[-1].z:+.1f})"       if ekfs  else "—"

        info_text.set_text(
            f"  Truth pos  {t_pos} m\n"
            f"  EKF pos    {e_pos} m\n"
            f"  ─────────────────────────────\n"
            f"  Position error  {err_str}\n"
            f"  Cov trace       {cov_str}\n"
            f"  Truth pts  {len(truths)}   EKF pts  {len(ekfs)}"
        )

        return truth_trail, truth_dot, ekf_trail, ekf_dot, error_line, info_text

    ani = animation.FuncAnimation(  # noqa: F841 — must stay alive
        fig,
        _update,
        interval=VIZ_UPDATE_MS,
        blit=False,
        cache_frame_data=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
