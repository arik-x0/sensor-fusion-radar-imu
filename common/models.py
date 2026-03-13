"""
Shared message schemas for the sensor-fusion microservice system.

All timestamps are UNIX epoch seconds (float).
All spatial quantities are in SI units:
  - distance: metres
  - velocity: m/s
  - acceleration: m/s²
  - angular rate: rad/s
  - angles: radians
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List
import msgpack


# ---------------------------------------------------------------------------
# Rover Ground Truth  (rover_service → radar_service, imu_service)
# ---------------------------------------------------------------------------

@dataclass
class RoverTruth:
    """True rover state published by the rover simulator.

    Both radar_service and imu_service subscribe to this topic and derive
    their respective noisy measurements from it, ensuring both sensors always
    observe the same physical rover.

    Accelerations are the rover's true kinematic (inertial) accelerations in
    the world frame — gravity is NOT included.  The IMU service converts
    these to body-frame specific force before adding sensor noise.
    """
    timestamp: float            # seconds since epoch

    # World-frame position (m)
    pos_x: float
    pos_y: float
    pos_z: float

    # World-frame velocity (m/s)
    vel_x: float
    vel_y: float
    vel_z: float

    # World-frame kinematic acceleration (m/s²)  — gravity excluded
    accel_world_x: float
    accel_world_y: float
    accel_world_z: float

    # Rover body orientation as a unit quaternion (w, x, y, z)
    # Encodes the rotation from body frame to world frame.
    qw: float
    qx: float
    qy: float
    qz: float

    # Body-frame angular rate (rad/s) — what the gyroscope measures (truth)
    gyro_x: float
    gyro_y: float
    gyro_z: float

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RoverTruth":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


# ---------------------------------------------------------------------------
# IMU Measurement
# ---------------------------------------------------------------------------

@dataclass
class ImuMeasurement:
    """Raw IMU packet published by the IMU service."""
    timestamp: float            # seconds since epoch
    accel_x: float              # m/s²
    accel_y: float
    accel_z: float
    gyro_x: float               # rad/s
    gyro_y: float
    gyro_z: float

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ImuMeasurement":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


# ---------------------------------------------------------------------------
# Radar Measurement
# ---------------------------------------------------------------------------

@dataclass
class RadarMeasurement:
    """Raw radar packet published by the radar service.

    Uses spherical coordinates relative to the radar sensor origin.
    """
    timestamp: float            # seconds since epoch
    range: float                # metres         (r ≥ 0)
    azimuth: float              # radians        (−π … π)
    elevation: float            # radians        (−π/2 … π/2)
    doppler: float              # m/s  (positive = approaching)

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RadarMeasurement":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


# ---------------------------------------------------------------------------
# EKF State Output
# ---------------------------------------------------------------------------

@dataclass
class EkfState:
    """Fused state estimate published by the EKF service after every update.

    Orientation is represented as a unit quaternion (w, x, y, z).
    Covariance is a flattened 10×10 row-major matrix (100 floats).
    """
    timestamp: float            # seconds since epoch

    # Position (m)
    pos_x: float
    pos_y: float
    pos_z: float

    # Velocity (m/s)
    vel_x: float
    vel_y: float
    vel_z: float

    # Orientation quaternion (unit)
    qw: float
    qx: float
    qy: float
    qz: float

    # Flattened 10×10 covariance matrix
    covariance: List[float] = field(default_factory=lambda: [0.0] * 100)

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EkfState":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def position(self):
        return (self.pos_x, self.pos_y, self.pos_z)

    @property
    def velocity(self):
        return (self.vel_x, self.vel_y, self.vel_z)

    @property
    def quaternion(self):
        return (self.qw, self.qx, self.qy, self.qz)


# ---------------------------------------------------------------------------
# Buffer Entry  (replay wrapper used by the buffer/backprop service)
# ---------------------------------------------------------------------------

@dataclass
class BufferEntry:
    """A source-tagged wrapper around a raw sensor measurement.

    The ``source`` field identifies the measurement type so the receiver can
    deserialise the ``payload`` bytes using the correct model class.

    source values:
        "imu"   → ImuMeasurement.from_bytes(payload)
        "radar" → RadarMeasurement.from_bytes(payload)

    is_late:
        True when the buffer service detected that this measurement arrived
        after messages with a later timestamp had already been released.
        The EKF service uses this flag to trigger backpropagation.
    """
    timestamp: float    # original measurement timestamp (for ordering)
    source: str         # "imu" | "radar"
    payload: bytes      # serialised ImuMeasurement or RadarMeasurement
    is_late: bool = False  # True if message arrived out-of-order

    def to_bytes(self) -> bytes:
        return msgpack.packb(
            {
                "timestamp": self.timestamp,
                "source": self.source,
                "payload": self.payload,
                "is_late": self.is_late,
            },
            use_bin_type=True,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BufferEntry":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            timestamp=d["timestamp"],
            source=d["source"],
            payload=d["payload"],
            is_late=d.get("is_late", False),
        )

    def unpack_measurement(self) -> "ImuMeasurement | RadarMeasurement":
        """Convenience: deserialise and return the inner measurement object."""
        if self.source == "imu":
            return ImuMeasurement.from_bytes(self.payload)
        elif self.source == "radar":
            return RadarMeasurement.from_bytes(self.payload)
        else:
            raise ValueError(f"Unknown source type: {self.source!r}")
