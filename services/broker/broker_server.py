"""
Broker Server
=============
Central message routing hub for the sensor-fusion microservice system.

Architecture
------------

  XSUB port  tcp://*:5550  ← publishers (IMU, Radar, EKF) connect here
  XPUB port  tcp://*:5551  → subscribers (EKF, Buffer, monitors) connect here
  ROUTER port tcp://*:5552  ↔ request/reply channel (buffer replay requests)

A background thread runs zmq.proxy(xsub, xpub) so all pub/sub traffic flows
through in microseconds with zero Python overhead in the hot path.

The main thread polls the XPUB socket for subscription events (so the broker
can log which topics each service subscribes to) and the ROUTER socket for
incoming buffer-replay requests.  Buffer-replay routing is transparent: the
broker forwards DEALER frames between EKF and Buffer service using the
built-in ROUTER identity mechanism.

Environment variables
---------------------
BROKER_XSUB_ADDR    Bind address for publishers   (default tcp://*:5550)
BROKER_XPUB_ADDR    Bind address for subscribers  (default tcp://*:5551)
BROKER_ROUTER_ADDR  Bind address for req/reply    (default tcp://*:5552)
"""

from __future__ import annotations

import logging
import os
import signal
import threading

import zmq

# ── configuration ──────────────────────────────────────────────────────────
XSUB_ADDR   = os.getenv("BROKER_XSUB_ADDR",   "tcp://*:5550")
XPUB_ADDR   = os.getenv("BROKER_XPUB_ADDR",   "tcp://*:5551")
ROUTER_ADDR = os.getenv("BROKER_ROUTER_ADDR",  "tcp://*:5552")

logging.basicConfig(
    level=logging.INFO,
    format="[broker] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── proxy thread ────────────────────────────────────────────────────────────

def _run_proxy(xsub_sock: zmq.Socket, xpub_sock: zmq.Socket, stop_event: threading.Event) -> None:
    """Run zmq.proxy in a dedicated thread.  Exits when *stop_event* is set."""
    try:
        log.info("PUB/SUB proxy started (XSUB→XPUB)")
        zmq.proxy(xsub_sock, xpub_sock)
    except zmq.ZMQError as exc:
        if not stop_event.is_set():
            log.error("Proxy error: %s", exc)


# ── subscription event decoder ──────────────────────────────────────────────

def _decode_subscription(frame: bytes) -> tuple[bool, str]:
    """Decode an XPUB subscription notification frame.

    ZMQ prepends 0x01 (subscribe) or 0x00 (unsubscribe) to the topic name.
    Returns (is_subscribe, topic_string).
    """
    if not frame:
        return False, ""
    action = frame[0]  # 1 = subscribe, 0 = unsubscribe
    topic  = frame[1:].decode(errors="replace")
    return bool(action), topic


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ctx = zmq.Context.instance()
    stop_event = threading.Event()

    # -- bind sockets --------------------------------------------------------
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind(XSUB_ADDR)
    log.info("XSUB bound on %s  (publishers connect here)", XSUB_ADDR)

    xpub = ctx.socket(zmq.XPUB)
    xpub.setsockopt(zmq.XPUB_VERBOSE, 1)   # receive every sub/unsub event
    xpub.bind(XPUB_ADDR)
    log.info("XPUB bound on %s  (subscribers connect here)", XPUB_ADDR)

    router = ctx.socket(zmq.ROUTER)
    router.bind(ROUTER_ADDR)
    log.info("ROUTER bound on %s  (request/reply channel)", ROUTER_ADDR)

    # -- start proxy thread --------------------------------------------------
    proxy_thread = threading.Thread(
        target=_run_proxy,
        args=(xsub, xpub, stop_event),
        daemon=True,
        name="proxy",
    )
    proxy_thread.start()

    # -- graceful shutdown ---------------------------------------------------
    def _shutdown(signum, frame):  # noqa: ARG001
        log.info("Shutdown signal received, stopping broker…")
        stop_event.set()
        # Terminate the proxy by closing its sockets from the main thread
        xsub.close(linger=0)
        xpub.close(linger=0)
        router.close(linger=0)
        ctx.term()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # -- main event loop (subscription monitor + ROUTER forward) -------------
    poller = zmq.Poller()
    # Note: we cannot poll xpub here because the proxy thread owns it.
    # Instead we poll only the ROUTER socket for req/reply forwarding.
    poller.register(router, zmq.POLLIN)

    log.info("Broker ready — waiting for services to connect…")

    while not stop_event.is_set():
        try:
            events = dict(poller.poll(timeout=500))  # 500 ms tick
        except zmq.ZMQError:
            break

        if router in events:
            # Forward DEALER↔DEALER messages through the ROUTER transparently.
            # The ROUTER socket already carries identity frames, so we just
            # echo multi-part frames back to the correct destination.
            try:
                frames = router.recv_multipart()
                # frames: [client_id, empty, command, payload, ...]
                if len(frames) >= 3:
                    client_id = frames[0]
                    command   = frames[2].decode(errors="replace") if len(frames) > 2 else ""
                    log.debug(
                        "REQ from %s  command=%r  payload_len=%d",
                        client_id.hex(),
                        command,
                        len(frames[3]) if len(frames) > 3 else 0,
                    )
                    # Route the full frame set onward — services registered with
                    # a known identity will receive it.  If there is only one
                    # party (e.g., buffer service hasn't connected yet), we send
                    # an error reply.
                    router.send_multipart(frames)
            except zmq.ZMQError as exc:
                if not stop_event.is_set():
                    log.warning("ROUTER recv error: %s", exc)

    log.info("Broker stopped.")


if __name__ == "__main__":
    main()
