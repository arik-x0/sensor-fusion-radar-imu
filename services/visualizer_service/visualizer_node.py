"""
Visualizer Service
==================
Subscribes to the EKF state topic and renders a live 3-D matplotlib plot
of the fused position trajectory.

Designed to run locally — requires a graphical display.
Not suitable for headless Docker without X11 forwarding / Xvfb.

Environment variables
---------------------
BROKER_XPUB_ADDR   ZMQ connect address for broker XPUB  (default tcp://localhost:5551)
VIZ_HISTORY        Number of past positions kept in the trail  (default 500)
VIZ_UPDATE_MS      Plot refresh interval in milliseconds       (default 100)
"""

from __future__ import annotations

import collections
import os
import threading

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import zmq
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3-D projection

from common.models import EkfState
from common.topics import TOPIC_STATE
from common.transport import Subscriber

BROKER_XPUB_ADDR = os.getenv("BROKER_XPUB_ADDR", "tcp://localhost:5551")
VIZ_HISTORY      = int(os.getenv("VIZ_HISTORY",    "500"))
VIZ_UPDATE_MS    = int(os.getenv("VIZ_UPDATE_MS",  "100"))


# ── Thread-safe state buffer ──────────────────────────────────────────────────

class _StateBuffer:
    """Ring buffer of EkfState objects shared between the subscriber thread and
    the matplotlib animation callback."""

    def __init__(self, maxlen: int) -> None:
        self._lock   = threading.Lock()
        self._states: collections.deque[EkfState] = collections.deque(maxlen=maxlen)

    def push(self, state: EkfState) -> None:
        with self._lock:
            self._states.append(state)

    def snapshot(self) -> list[EkfState]:
        """Return a shallow copy of current contents (safe to iterate off-thread)."""
        with self._lock:
            return list(self._states)


# ── Subscriber thread ─────────────────────────────────────────────────────────

def _subscriber_thread(buf: _StateBuffer) -> None:
    sub = Subscriber(
        addresses=[BROKER_XPUB_ADDR],
        topics=[TOPIC_STATE],
        recv_timeout_ms=2000,
    )
    while True:
        try:
            _, payload = sub.recv()
            buf.push(EkfState.from_bytes(payload))
        except zmq.Again:
            pass
        except Exception:
            pass


# ── Matplotlib live plot ──────────────────────────────────────────────────────

def _build_figure():
    """Create and return (fig, ax, artists) for the 3-D trajectory plot."""
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle("EKF 3-D Position Trajectory", fontsize=13, fontweight="bold")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Trail line — full history
    (trail_line,) = ax.plot([], [], [], "-", color="steelblue", lw=1.5,
                            alpha=0.65, label="Trajectory")

    # Current position marker
    (current_dot,) = ax.plot([], [], [], "o", color="crimson", ms=9,
                             label="Current position")

    # Velocity vector (quiver) — updated each frame; stored in a list so the
    # closure can replace the artist without a nonlocal statement.
    vel_quiver: list = [None]

    # Info text box (2-D overlay in axes coordinates)
    info_text = ax.text2D(
        0.02, 0.97, "",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.85),
    )

    ax.legend(loc="upper right", fontsize=8)

    return fig, ax, trail_line, current_dot, vel_quiver, info_text


def main():
    buf = _StateBuffer(maxlen=VIZ_HISTORY)

    # Start the subscriber on a daemon thread so it doesn't block on exit.
    threading.Thread(
        target=_subscriber_thread,
        args=(buf,),
        daemon=True,
        name="ekf-sub",
    ).start()

    fig, ax, trail_line, current_dot, vel_quiver, info_text = _build_figure()

    def _update(_frame):
        states = buf.snapshot()
        if not states:
            return trail_line, current_dot, info_text

        xs = np.array([s.pos_x for s in states])
        ys = np.array([s.pos_y for s in states])
        zs = np.array([s.pos_z for s in states])

        # ── Trail ──────────────────────────────────────────────────────────
        trail_line.set_data(xs, ys)
        trail_line.set_3d_properties(zs)

        # ── Current position ────────────────────────────────────────────────
        current_dot.set_data([xs[-1]], [ys[-1]])
        current_dot.set_3d_properties([zs[-1]])

        # ── Velocity arrow (remove previous, then redraw) ───────────────────
        if vel_quiver[0] is not None:
            vel_quiver[0].remove()
        last = states[-1]
        vel_quiver[0] = ax.quiver(
            last.pos_x, last.pos_y, last.pos_z,
            last.vel_x, last.vel_y, last.vel_z,
            color="darkorange", linewidth=2, arrow_length_ratio=0.3,
        )

        # ── Axis limits with padding ────────────────────────────────────────
        pad = max(2.0, (xs.ptp() + ys.ptp() + zs.ptp()) * 0.1)
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)
        ax.set_zlim(zs.min() - pad, zs.max() + pad)

        # ── Info overlay ────────────────────────────────────────────────────
        cov_trace = float(np.trace(np.array(last.covariance).reshape(10, 10)))
        info_text.set_text(
            f"t = {last.timestamp:.3f} s\n"
            f"pos  ({last.pos_x:+.2f}, {last.pos_y:+.2f}, {last.pos_z:+.2f}) m\n"
            f"vel  ({last.vel_x:+.2f}, {last.vel_y:+.2f}, {last.vel_z:+.2f}) m/s\n"
            f"cov trace  {cov_trace:.3e}\n"
            f"history  {len(states)} / {VIZ_HISTORY}"
        )

        return trail_line, current_dot, info_text

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
