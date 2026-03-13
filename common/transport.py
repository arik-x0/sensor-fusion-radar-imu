"""
ZeroMQ transport helpers for the sensor-fusion broker architecture.

All services connect TO the broker rather than binding their own ports.
The broker itself uses the bind-side counterparts (XPUB/XSUB/ROUTER).

Classes
-------
Publisher   — PUB socket that CONNECTs to the broker's XSUB port.
Subscriber  — SUB socket that CONNECTs to the broker's XPUB port.
Requester   — DEALER socket for async request/reply to the broker ROUTER.
Replier     — ROUTER socket server-side wrapper (used by the buffer service
              via the broker; direct use is optional).
"""

from __future__ import annotations

import zmq


# ---------------------------------------------------------------------------
# Publisher  (connects to broker XSUB port)
# ---------------------------------------------------------------------------

class Publisher:
    """A ZeroMQ PUB socket wrapper.

    Parameters
    ----------
    address:
        Either a bind address (e.g. ``"tcp://*:5555"``) when ``mode="bind"``
        or a connect address (e.g. ``"tcp://localhost:5550"``) when
        ``mode="connect"`` (default for sensor services connecting to the broker).
    mode:
        ``"bind"`` or ``"connect"``.  Services should use ``"connect"``;
        only the broker (XSUB side) uses ``"bind"``.
    """

    def __init__(self, address: str, mode: str = "connect") -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        if mode == "bind":
            self._sock.bind(address)
        else:
            self._sock.connect(address)

    def send(self, topic: str, payload: bytes) -> None:
        """Send *payload* bytes prefixed with *topic*."""
        self._sock.send_multipart([topic.encode(), payload])

    def close(self) -> None:
        self._sock.close()

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Subscriber  (connects to broker XPUB port)
# ---------------------------------------------------------------------------

class Subscriber:
    """A ZeroMQ SUB socket wrapper.

    Parameters
    ----------
    addresses:
        One or more ``tcp://host:port`` strings to connect to.
    topics:
        Topics to subscribe to (empty string = all).
    recv_timeout_ms:
        How long `.recv()` blocks waiting for a message, in ms.
        ``-1`` means block forever.
    """

    def __init__(
        self,
        addresses: list[str],
        topics: list[str],
        recv_timeout_ms: int = -1,
    ) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        for addr in addresses:
            self._sock.connect(addr)
        for topic in topics:
            self._sock.setsockopt(zmq.SUBSCRIBE, topic.encode())

    def recv(self) -> tuple[str, bytes]:
        """Block until a message arrives; return ``(topic, payload_bytes)``.

        Raises ``zmq.Again`` if the receive timeout was hit.
        """
        parts = self._sock.recv_multipart()
        topic = parts[0].decode()
        payload = parts[1]
        return topic, payload

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Requester  (DEALER — async request/reply client, used by EKF → broker)
# ---------------------------------------------------------------------------

class Requester:
    """A ZeroMQ DEALER socket for sending requests to the broker ROUTER.

    Unlike a plain REQ socket, DEALER is non-blocking and supports fire-and-
    forget semantics.  Callers must include an empty-frame delimiter manually.

    Parameters
    ----------
    address:
        Broker ROUTER address, e.g. ``"tcp://localhost:5552"``.
    identity:
        Optional socket identity bytes (for routing replies back correctly).
    recv_timeout_ms:
        Timeout for `.recv_reply()`.  ``-1`` = block forever.
    """

    def __init__(
        self,
        address: str,
        identity: bytes | None = None,
        recv_timeout_ms: int = 5000,
    ) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        if identity:
            self._sock.setsockopt(zmq.IDENTITY, identity)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._sock.connect(address)

    def send_request(self, command: str, payload: bytes = b"") -> None:
        """Send a request frame: [empty | command | payload]."""
        self._sock.send_multipart([b"", command.encode(), payload])

    def recv_reply(self) -> tuple[str, bytes]:
        """Receive a reply frame: [empty | status | payload].

        Returns ``(status_str, payload_bytes)``.
        Raises ``zmq.Again`` on timeout.
        """
        parts = self._sock.recv_multipart()
        # parts: [empty, status, payload]
        status = parts[1].decode() if len(parts) > 1 else ""
        payload = parts[2] if len(parts) > 2 else b""
        return status, payload

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Replier  (ROUTER — request/reply server, used directly by buffer service)
# ---------------------------------------------------------------------------

class Replier:
    """A ZeroMQ ROUTER socket for receiving and replying to requests.

    Used by the Buffer service to handle replay requests from EKF / other
    services.  In the broker architecture the Buffer service connects its
    own ROUTER to the broker's DEALER-facing port.

    Parameters
    ----------
    address:
        Bind address, e.g. ``"tcp://*:5552"``.
    recv_timeout_ms:
        Timeout for `.recv_request()`.  ``-1`` = block forever.
    """

    def __init__(self, address: str, recv_timeout_ms: int = 100) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._sock.bind(address)

    def recv_request(self) -> tuple[bytes, str, bytes]:
        """Receive next request.

        Returns ``(client_id, command, payload)``.
        Raises ``zmq.Again`` on timeout.
        """
        parts = self._sock.recv_multipart()
        client_id = parts[0]
        command   = parts[2].decode() if len(parts) > 2 else ""
        payload   = parts[3] if len(parts) > 3 else b""
        return client_id, command, payload

    def send_reply(self, client_id: bytes, status: str, payload: bytes = b"") -> None:
        """Send a reply back to *client_id*."""
        self._sock.send_multipart([client_id, b"", status.encode(), payload])

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
