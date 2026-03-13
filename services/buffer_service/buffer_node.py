"""
Buffer Service
==============
Synchronizes and time-orders radar and IMU sensor measurements before
forwarding them to the EKF service.  Acts as a jitter-absorbing relay
between the raw sensor publishers and the EKF consumer.

Responsibilities
----------------
1. Subscribe to TOPIC_IMU (100 Hz) and TOPIC_RADAR (10 Hz) from the broker.
2. Hold incoming messages in a min-heap sorted by measurement timestamp.
3. Every BUFFER_WINDOW_MS milliseconds, flush all messages older than the
   current wall-clock time minus the window, publishing them as BufferEntry
   objects on TOPIC_BUFFER in strict timestamp order.
4. Detect late-arriving messages (timestamp < last flushed timestamp) and
   mark BufferEntry.is_late = True so the EKF can trigger backpropagation.

Backpropagation protocol
------------------------
When the EKF receives a BufferEntry with is_late=True it:
  a) Looks up the state snapshot closest to (but before) the late timestamp.
  b) Resets the EKF to that snapshot.
  c) Replays all subsequent buffered measurements from its internal history.

The buffer service itself does not perform EKF backprop; it only flags
the anomaly and lets the EKF handle its own state history.

Environment variables
---------------------
BROKER_XPUB_ADDR    Broker subscriber port  (default tcp://broker_service:5551)
BROKER_XSUB_ADDR    Broker publisher port   (default tcp://broker_service:5550)
BUFFER_WINDOW_MS    Sync window in ms       (default 100)
BUFFER_MAX_AGE_S    Drop messages older than this many seconds (default 5.0)
"""

import heapq
import logging
import os
import threading
import time

import zmq

from common.models import BufferEntry, ImuMeasurement, RadarMeasurement
from common.topics import TOPIC_BUFFER, TOPIC_IMU, TOPIC_RADAR
from common.transport import Publisher, Subscriber

# ── configuration ──────────────────────────────────────────────────────────
BROKER_XPUB_ADDR = os.getenv("BROKER_XPUB_ADDR", "tcp://broker_service:5551")
BROKER_XSUB_ADDR = os.getenv("BROKER_XSUB_ADDR", "tcp://broker_service:5550")
BUFFER_WINDOW_MS = float(os.getenv("BUFFER_WINDOW_MS", "100"))   # ms
BUFFER_MAX_AGE_S = float(os.getenv("BUFFER_MAX_AGE_S", "5.0"))   # seconds

logging.basicConfig(
    level=logging.INFO,
    format="[buffer_service] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Buffer node ─────────────────────────────────────────────────────────────

class BufferNode:
    """
    Receives raw sensor messages, holds them in a sorted min-heap, then
    flushes them to TOPIC_BUFFER in timestamp order every BUFFER_WINDOW_MS ms.

    Min-heap entries: (timestamp, counter, source, raw_bytes)
    The monotonic ``counter`` is a tie-breaker so that two messages with
    identical timestamps have a stable, well-defined heap order without
    falling back to comparing bytes objects.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str, bytes]] = []
        self._counter = 0
        self._lock = threading.Lock()
        self._last_flushed_ts: float = 0.0
        self._pub = Publisher(BROKER_XSUB_ADDR)
        self._late_count = 0
        self._flush_count = 0

        log.info("Buffer service starting…")
        log.info("  Subscribe ← %s  (topics: %s, %s)", BROKER_XPUB_ADDR, TOPIC_IMU, TOPIC_RADAR)
        log.info("  Publish   → %s  (topic: %s)", BROKER_XSUB_ADDR, TOPIC_BUFFER)
        log.info("  Window: %.0f ms  |  Max age: %.1f s", BUFFER_WINDOW_MS, BUFFER_MAX_AGE_S)

    # ------------------------------------------------------------------
    # Sensor receiver threads
    # ------------------------------------------------------------------

    def _recv_loop(self, source: str, topic: str) -> None:
        """Receive messages for one sensor topic and push onto the heap."""
        sub = Subscriber(
            addresses=[BROKER_XPUB_ADDR],
            topics=[topic],
            recv_timeout_ms=1000,
        )
        log.info("[%s] receiver ready", source)
        while True:
            try:
                _, raw = sub.recv()
                # Decode only enough to extract the timestamp.
                if source == "imu":
                    ts = ImuMeasurement.from_bytes(raw).timestamp
                else:
                    ts = RadarMeasurement.from_bytes(raw).timestamp

                with self._lock:
                    heapq.heappush(
                        self._heap,
                        (ts, self._counter, source, raw),
                    )
                    self._counter += 1

            except zmq.Again:
                pass   # timeout — keep waiting
            except Exception as exc:
                log.error("[%s] recv error: %s", source, exc)

    # ------------------------------------------------------------------
    # Flush thread — publishes sorted messages on TOPIC_BUFFER
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        """Every BUFFER_WINDOW_MS ms, flush aged-out messages in timestamp order."""
        window_s = BUFFER_WINDOW_MS / 1000.0
        log.info("Flush loop started (%.0f ms window)", BUFFER_WINDOW_MS)

        while True:
            time.sleep(window_s)
            now = time.time()
            cutoff = now - window_s   # release only messages older than the window

            batch: list[tuple[float, int, str, bytes]] = []
            with self._lock:
                while self._heap:
                    ts, cnt, source, raw = self._heap[0]
                    age = now - ts
                    if age > BUFFER_MAX_AGE_S:
                        heapq.heappop(self._heap)
                        log.debug("Dropped stale %s message (age=%.2fs)", source, age)
                        continue
                    if ts <= cutoff:
                        heapq.heappop(self._heap)
                        batch.append((ts, cnt, source, raw))
                    else:
                        break   # min-heap: all remaining entries are newer

            for ts, _, source, raw in batch:
                is_late = ts < self._last_flushed_ts
                if is_late:
                    self._late_count += 1
                    log.warning(
                        "Late %s message: ts=%.3f  last_flushed=%.3f  "
                        "delay=%.3fs",
                        source, ts, self._last_flushed_ts,
                        self._last_flushed_ts - ts,
                    )

                entry = BufferEntry(
                    timestamp=ts,
                    source=source,
                    payload=raw,
                    is_late=is_late,
                )
                try:
                    self._pub.send(TOPIC_BUFFER, entry.to_bytes())
                except Exception as exc:
                    log.error("Publish error: %s", exc)

                self._last_flushed_ts = max(self._last_flushed_ts, ts)
                self._flush_count += 1

            # Periodic stats log every 500 published messages
            if self._flush_count > 0 and self._flush_count % 500 == 0:
                log.info(
                    "Stats — flushed=%d  late=%d  heap_size=%d",
                    self._flush_count,
                    self._late_count,
                    len(self._heap),
                )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        t_imu = threading.Thread(
            target=self._recv_loop, args=("imu", TOPIC_IMU),
            daemon=True, name="imu-recv",
        )
        t_radar = threading.Thread(
            target=self._recv_loop, args=("radar", TOPIC_RADAR),
            daemon=True, name="radar-recv",
        )
        t_flush = threading.Thread(
            target=self._flush_loop,
            daemon=True, name="flush",
        )

        t_imu.start()
        t_radar.start()
        t_flush.start()

        log.info("Buffer Service running — synchronising sensor streams…")
        try:
            t_flush.join()
        except KeyboardInterrupt:
            log.info("Buffer Service shutting down")
        finally:
            self._pub.close()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    node = BufferNode()
    node.run()


if __name__ == "__main__":
    main()
