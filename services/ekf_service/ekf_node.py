"""
EKF Service — Main Service Node
================================
Subscribes to the TOPIC_BUFFER stream produced by the buffer service,
dispatches each BufferEntry to the EKF predict (IMU) or update (radar)
step, and publishes the fused state estimate after every radar update.

Pipeline
--------
  Buffer (sorted) → BufferEntry
    source=="imu"   → ekf.predict()
    source=="radar" → ekf.update() → publish EkfState

Backpropagation
---------------
When a BufferEntry arrives with is_late=True the node:
  1. Finds the state snapshot taken just before the late measurement's
     timestamp.
  2. Restores the EKF to that snapshot.
  3. Inserts the late measurement at its correct position and replays all
     subsequent measurements from the internal history buffer.

This retroactively corrects the EKF trajectory for delayed sensor data
without requiring any external replay requests.

Environment variables
---------------------
BROKER_XPUB_ADDR   ZMQ connect address for broker XPUB  (default tcp://broker_service:5551)
BROKER_XSUB_ADDR   ZMQ connect address for broker XSUB  (default tcp://broker_service:5550)

EKF noise parameters (all floats):
  EKF_SIGMA_ACCEL      accelerometer noise std-dev (m/s²)       default 0.1
  EKF_SIGMA_GYRO       gyro noise std-dev (rad/s)               default 0.01
  EKF_SIGMA_RANGE      radar range noise (m)                    default 2.0
  EKF_SIGMA_ANGLE      radar angle noise (rad)                  default 0.02
  EKF_SIGMA_DOPPLER    radar Doppler noise (m/s)                default 0.5
  EKF_SIGMA_ABIAS_RW   accel bias random-walk noise (m/s²/√s)  default 1e-4
  EKF_SIGMA_GBIAS_RW   gyro bias random-walk noise (rad/s/√s)  default 1e-5
  EKF_INNOV_GATE       innovation gate (normalised y²/S)        default 5.0
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time

import zmq

from common.models import BufferEntry, EkfState, ImuMeasurement, RadarMeasurement
from common.topics import TOPIC_BUFFER, TOPIC_STATE
from common.transport import Publisher, Subscriber
from ekf_core import ExtendedKalmanFilter

# ── configuration ──────────────────────────────────────────────────────────
BROKER_XPUB_ADDR  = os.getenv("BROKER_XPUB_ADDR",  "tcp://broker_service:5551")
BROKER_XSUB_ADDR  = os.getenv("BROKER_XSUB_ADDR",  "tcp://broker_service:5550")

EKF_SIGMA_ACCEL   = float(os.getenv("EKF_SIGMA_ACCEL",   "0.1"))
EKF_SIGMA_GYRO    = float(os.getenv("EKF_SIGMA_GYRO",    "0.01"))
EKF_SIGMA_RANGE   = float(os.getenv("EKF_SIGMA_RANGE",   "2.0"))
EKF_SIGMA_ANGLE   = float(os.getenv("EKF_SIGMA_ANGLE",   "0.02"))
EKF_SIGMA_DOPPLER = float(os.getenv("EKF_SIGMA_DOPPLER", "0.5"))
EKF_SIGMA_ABIAS_RW = float(os.getenv("EKF_SIGMA_ABIAS_RW", "1e-4"))
EKF_SIGMA_GBIAS_RW = float(os.getenv("EKF_SIGMA_GBIAS_RW", "1e-5"))
EKF_INNOV_GATE     = float(os.getenv("EKF_INNOV_GATE",     "5.0"))

# Number of state snapshots to keep (≈ 20 s of radar updates at 10 Hz)
MAX_SNAPSHOTS = 200

logging.basicConfig(
    level=logging.INFO,
    format="[ekf_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── State snapshot ───────────────────────────────────────────────────────────

class _Snapshot:
    """Lightweight copy of EKF state/covariance taken after each radar update."""
    __slots__ = ("timestamp", "x", "P", "last_time")

    def __init__(self, timestamp: float, ekf: ExtendedKalmanFilter) -> None:
        self.timestamp = timestamp
        self.x         = ekf.x.copy()
        self.P         = ekf.P.copy()
        self.last_time = ekf._last_time


# ── EKF service orchestrator ─────────────────────────────────────────────────

class EkfServiceNode:
    """
    Consumes time-ordered BufferEntry messages from the buffer service and
    feeds them into the EKF predict/update pipeline.

    Thread safety
    -------------
    A single subscriber thread owns the EKF; no additional locking is needed
    because all EKF mutations happen on that thread.

    Backpropagation
    ---------------
    After every radar update a _Snapshot is stored.  When a late message
    arrives, the node restores the nearest earlier snapshot and replays all
    subsequent messages from the internal history deque.
    """

    def __init__(self) -> None:
        self._ekf = ExtendedKalmanFilter(
            sigma_accel=EKF_SIGMA_ACCEL,
            sigma_gyro=EKF_SIGMA_GYRO,
            sigma_range=EKF_SIGMA_RANGE,
            sigma_angle=EKF_SIGMA_ANGLE,
            sigma_doppler=EKF_SIGMA_DOPPLER,
            sigma_abias_rw=EKF_SIGMA_ABIAS_RW,
            sigma_gbias_rw=EKF_SIGMA_GBIAS_RW,
            innov_gate=EKF_INNOV_GATE,
        )
        # Circular snapshot buffer (indexed by radar updates)
        self._snapshots: collections.deque[_Snapshot] = collections.deque(
            maxlen=MAX_SNAPSHOTS
        )
        # Raw measurement history for replay: (timestamp, source, payload_bytes)
        self._history: collections.deque[tuple[float, str, bytes]] = collections.deque(
            maxlen=MAX_SNAPSHOTS * 12   # 12× because IMU is 10× faster than radar
        )

        self._pub = Publisher(BROKER_XSUB_ADDR)

        log.info("EKF Service starting…")
        log.info("  Subscribe ← %s  (topic: %s)", BROKER_XPUB_ADDR, TOPIC_BUFFER)
        log.info("  Publish   → %s  (topic: %s)", BROKER_XSUB_ADDR, TOPIC_STATE)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _save_snapshot(self, ts: float) -> None:
        self._snapshots.append(_Snapshot(ts, self._ekf))

    def _find_snapshot_before(self, ts: float) -> _Snapshot | None:
        """Return the most recent snapshot with timestamp ≤ ts, or None."""
        best: _Snapshot | None = None
        for snap in self._snapshots:
            if snap.timestamp <= ts:
                if best is None or snap.timestamp > best.timestamp:
                    best = snap
        return best

    def _restore_snapshot(self, snap: _Snapshot) -> None:
        self._ekf.x         = snap.x.copy()
        self._ekf.P         = snap.P.copy()
        self._ekf._last_time = snap.last_time
        log.info("Restored EKF snapshot @ t=%.3f", snap.timestamp)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, late_entry: BufferEntry) -> None:
        """Rewind EKF to the snapshot before the late entry and replay."""
        late_ts = late_entry.timestamp
        snap = self._find_snapshot_before(late_ts)
        if snap is None:
            log.warning(
                "Backprop: no snapshot found before ts=%.3f — processing in-order",
                late_ts,
            )
            # Fall through and process the late message as-is
            self._apply_entry(late_entry.source, late_entry.payload)
            return

        log.info(
            "Backprop triggered: late_ts=%.3f  snap_ts=%.3f  "
            "rewind=%.3fs",
            late_ts, snap.timestamp, late_ts - snap.timestamp,
        )
        self._restore_snapshot(snap)

        # Build a sorted replay list: late entry + all history after the snapshot
        replay: list[tuple[float, str, bytes]] = [
            (late_ts, late_entry.source, late_entry.payload)
        ]
        for h_ts, h_source, h_raw in self._history:
            if h_ts > snap.timestamp:
                replay.append((h_ts, h_source, h_raw))
        replay.sort(key=lambda t: t[0])

        log.info("Backprop: replaying %d measurements", len(replay))
        for ts, source, raw in replay:
            self._apply_entry(source, raw, snapshot_ts=ts if source == "radar" else None)

    def _apply_entry(
        self,
        source: str,
        raw: bytes,
        snapshot_ts: float | None = None,
    ) -> None:
        """Apply one raw measurement to the EKF; publish + snapshot on radar."""
        if source == "imu":
            imu = ImuMeasurement.from_bytes(raw)
            self._ekf.predict(imu)
        else:
            radar = RadarMeasurement.from_bytes(raw)
            self._ekf.update(radar)

            now = time.time()
            state = self._ekf.get_state(timestamp=now)
            self._pub.send(TOPIC_STATE, state.to_bytes())
            self._save_snapshot(snapshot_ts if snapshot_ts is not None else now)

            log.info(
                "State published  pos=(%.2f, %.2f, %.2f) m  "
                "vel=(%.2f, %.2f, %.2f) m/s",
                state.pos_x, state.pos_y, state.pos_z,
                state.vel_x, state.vel_y, state.vel_z,
            )

    # ------------------------------------------------------------------
    # Buffer subscriber loop
    # ------------------------------------------------------------------

    def _buffer_loop(self) -> None:
        sub = Subscriber(
            addresses=[BROKER_XPUB_ADDR],
            topics=[TOPIC_BUFFER],
            recv_timeout_ms=2000,
        )
        log.info("Buffer subscriber ready")

        while True:
            try:
                _, raw = sub.recv()
                entry = BufferEntry.from_bytes(raw)

                if entry.is_late:
                    self._backpropagate(entry)
                else:
                    self._apply_entry(entry.source, entry.payload)

                # Keep measurement in history for future backprop replays
                self._history.append((entry.timestamp, entry.source, entry.payload))

            except zmq.Again:
                log.warning("No buffer message received within timeout")
            except Exception as exc:
                log.error("Buffer loop error: %s", exc)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        t_buf = threading.Thread(
            target=self._buffer_loop, daemon=True, name="buf-sub"
        )
        t_buf.start()

        log.info("EKF Service running — waiting for buffer data…")
        try:
            t_buf.join()
        except KeyboardInterrupt:
            log.info("EKF Service shutting down")
        finally:
            self._pub.close()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    node = EkfServiceNode()
    node.run()


if __name__ == "__main__":
    main()
