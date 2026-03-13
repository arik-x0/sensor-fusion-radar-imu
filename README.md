# Sensor Fusion — Radar + IMU with EKF

A **microservice-based sensor fusion system** that tracks a target in 3-D space
by fusing radar and IMU measurements through an Extended Kalman Filter (EKF).

---

## Architecture

All inter-service messaging is routed through a central **broker** using ZeroMQ
XSUB/XPUB sockets. A dedicated **buffer service** sits between the raw sensors
and the EKF to synchronize the two streams and detect late-arriving messages.

```
┌─────────────────┐                                      ┌─────────────────┐
│  radar_service  │──PUB (topic: radar)──▶┐              │ monitor_service  │
│  10 Hz          │                       │              │ prints state     │
└─────────────────┘                       ▼              └────────▲────────┘
                                  ┌───────────────┐               │
┌─────────────────┐               │               │               │
│  imu_service    │──PUB (imu)──▶ │ broker_service│──XPUB──▶──────┤
│  100 Hz         │               │  XSUB / XPUB  │               │
└─────────────────┘               │  ROUTER       │          ┌────┴────────────────┐
                                  └───────────────┘          │   ekf_service       │
                                          │                  │                     │
                                    XPUB  │                  │  predict()  (IMU)   │
                                          ▼                  │  update()   (Radar) │
                                  ┌───────────────┐          │  backprop on late   │
                                  │ buffer_service│──buffer──▶  messages           │
                                  │  sync + order │          └─────────────────────┘
                                  │  late detect  │
                                  └───────────────┘
```

### Services

| Service | Role |
|---|---|
| `broker_service` | Central message router (XSUB/XPUB/ROUTER). All publishers and subscribers connect through it. |
| `radar_service` | Simulates target radar returns — publishes `RadarMeasurement` at 10 Hz |
| `imu_service` | Simulates body-frame IMU — publishes `ImuMeasurement` at 100 Hz |
| `buffer_service` | Buffers and time-orders both sensor streams; flags late messages for backpropagation |
| `ekf_service` | **Main service** — runs EKF predict/update, publishes `EkfState`; performs backpropagation on late data |
| `monitor_service` | Subscribes to EKF state, pretty-prints position/velocity/attitude and covariance trace |

### Broker Ports

| Port | Socket | Purpose |
|---|---|---|
| `5550` | XSUB | Publishers (radar, imu, ekf) connect here |
| `5551` | XPUB | Subscribers (buffer, ekf, monitor) connect here |
| `5552` | ROUTER | Request/reply channel (buffer replay requests) |

### EKF State Vector

| Index | Symbol | Description |
|---|---|---|
| 0–2 | `p` | Position (x, y, z) in metres |
| 3–5 | `v` | Velocity (x, y, z) in m/s |
| 6–9 | `q` | Orientation quaternion (w, x, y, z) |

---

## Message Flow

1. **radar_service** and **imu_service** publish raw measurements to the broker.
2. **buffer_service** subscribes to both topics, holds messages in a min-heap sorted by timestamp, and flushes them every `BUFFER_WINDOW_MS` in strict timestamp order as `BufferEntry` packets on the `buffer` topic.
3. If a message arrives after messages with a later timestamp have already been released, `BufferEntry.is_late` is set to `True`.
4. **ekf_service** subscribes only to `TOPIC_BUFFER`. It dispatches each entry to `ekf.predict()` (IMU) or `ekf.update()` (radar) and publishes the fused `EkfState` after every radar update.
5. On `is_late=True`, the EKF rewinds to the stored state snapshot just before the late measurement's timestamp, inserts the late measurement, and replays all subsequent measurements from its internal history buffer (backpropagation).
6. **monitor_service** subscribes to the fused state and prints it.

---

## Quick Start

### With Docker (recommended)

```bash
# Build and run all services
docker compose up --build

# Watch the fused state output
docker compose logs -f monitor_service

# Watch buffer sync stats
docker compose logs -f buffer_service

# Watch the broker routing
docker compose logs -f broker_service
```

### Without Docker (local Python)

Run each service in a separate terminal. Start the broker first.

```bash
# Install dependencies
pip install pyzmq msgpack numpy

# Set PYTHONPATH so imports resolve
set PYTHONPATH=.          # Windows
export PYTHONPATH=.       # Linux/macOS

# 1 — Broker (must start first)
python services/broker/broker_server.py

# 2 — Sensors (order does not matter)
python services/radar_service/radar_node.py
python services/imu_service/imu_node.py

# 3 — Buffer (after sensors are up)
python services/buffer_service/buffer_node.py

# 4 — EKF (after buffer is up)
python services/ekf_service/ekf_node.py

# 5 — Monitor
python services/monitor_service/monitor_node.py
```

When running locally, override the default broker address if needed:

```bash
export BROKER_XSUB_ADDR=tcp://localhost:5550
export BROKER_XPUB_ADDR=tcp://localhost:5551
```

---

## Configuration

All parameters are controlled via environment variables (see `.env`).

| Variable | Default | Description |
|---|---|---|
| `RADAR_HZ` | `10` | Radar publication rate (Hz) |
| `RADAR_NOISE_RANGE` | `1.5` | Range noise std-dev (m) |
| `RADAR_NOISE_ANGLE` | `0.01` | Azimuth/elevation noise std-dev (rad) |
| `RADAR_NOISE_DOPPLER` | `0.5` | Doppler noise std-dev (m/s) |
| `IMU_HZ` | `100` | IMU publication rate (Hz) |
| `IMU_NOISE_ACCEL` | `0.05` | Accel noise std-dev (m/s²) |
| `IMU_NOISE_GYRO` | `0.005` | Gyro noise std-dev (rad/s) |
| `BUFFER_WINDOW_MS` | `100` | Buffer sync window — messages older than this are released (ms) |
| `BUFFER_MAX_AGE_S` | `5.0` | Messages older than this are dropped as stale (s) |
| `EKF_SIGMA_ACCEL` | `0.1` | EKF process noise — accel |
| `EKF_SIGMA_GYRO` | `0.01` | EKF process noise — gyro |
| `EKF_SIGMA_RANGE` | `2.0` | EKF measurement noise — range |
| `EKF_SIGMA_ANGLE` | `0.02` | EKF measurement noise — angles |
| `EKF_SIGMA_DOPPLER` | `0.5` | EKF measurement noise — Doppler |

---

## Unit Tests

```bash
pip install pytest numpy msgpack
python -m pytest services/ekf_service/tests/ -v
```

Tests cover:
- Predict: covariance growth, position propagation, quaternion unit-norm, stale packet rejection
- Update: covariance reduction, state convergence toward measurement, PSD guarantee, long-run numerical stability

---

## Project Structure

```
sensor-fusion-radar-imu/
├── common/                        # Shared package (all services import this)
│   ├── models.py                  # ImuMeasurement, RadarMeasurement, EkfState, BufferEntry
│   ├── transport.py               # ZeroMQ Publisher / Subscriber / Requester / Replier
│   └── topics.py                  # Topic name constants
│
├── services/
│   ├── broker/                    # Central message router
│   │   ├── broker_server.py       # XSUB/XPUB proxy + ROUTER request/reply
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── radar_service/
│   │   ├── radar_node.py          # Radar simulator — publishes to broker
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── imu_service/
│   │   ├── imu_node.py            # IMU simulator — publishes to broker
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── buffer_service/            # ← sync + backprop layer
│   │   ├── buffer_node.py         # Min-heap time-ordered relay with late detection
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── ekf_service/               # ← main fusion service
│   │   ├── ekf_core.py            # EKF mathematics (predict + update)
│   │   ├── ekf_node.py            # Consumes TOPIC_BUFFER; backprop on late messages
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── tests/
│   │       └── test_ekf_core.py
│   │
│   └── monitor_service/
│       ├── monitor_node.py        # State display — subscribes via broker
│       ├── Dockerfile
│       └── requirements.txt
│
├── docker-compose.yml
├── .env                           # Default parameter values
└── README.md
```
