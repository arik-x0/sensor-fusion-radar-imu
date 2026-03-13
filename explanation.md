# What Is This System and Why Does It Exist?

## The Core Problem

Imagine you are trying to track a moving object — an aircraft, a drone, a vehicle — in 3-D space. You have two sensors available:

| Sensor | What it measures | Strengths | Weaknesses |
|---|---|---|---|
| **Radar** | Range, angles, and Doppler velocity to the target | Directly observes absolute position | Slow (10 Hz), noisy, no orientation info |
| **IMU** | Body-frame acceleration and angular rate | Very fast (100 Hz), smooth, tracks rotation | Drifts over time — errors accumulate with every integration |

Neither sensor alone gives you a reliable, complete picture. Radar is too noisy and too slow. An IMU alone will drift away from the truth within seconds because integrating noisy acceleration compounds errors.

**Sensor fusion** is the technique of combining both sensors so their weaknesses cancel each other out and their strengths reinforce each other.

---

## What is Built

A **real-time, microservice-based sensor fusion pipeline** that continuously tracks a target in 3-D space by fusing radar and IMU data through an **Extended Kalman Filter (EKF)**.

The system runs as six independent services that communicate over a message bus (ZeroMQ):

```
radar (10 Hz) ──▶
                  broker ──▶ buffer ──▶ EKF ──▶ monitor (stdout)
imu (100 Hz)  ──▶                              ──▶ visualizer (3-D plot)
```

### The Six Services

| Service | What it does |
|---|---|
| **broker** | Central message router. All services publish and subscribe through it — no service talks directly to another. |
| **radar_service** | Simulates a target moving through 3-D space and emits noisy radar measurements (range, azimuth, elevation, Doppler) at 10 Hz. |
| **imu_service** | Simulates an IMU on the tracking platform, emitting noisy body-frame accelerations and angular rates at 100 Hz. |
| **buffer_service** | Receives both streams, sorts them into strict timestamp order using a min-heap, and flags any message that arrived late (out-of-order). |
| **ekf_service** | The heart of the system. Runs the EKF predict/update cycle and publishes the fused state estimate after every radar update. |
| **monitor_service** | Subscribes to the fused state and prints position, velocity, attitude, and uncertainty to the terminal. |
| **visualizer_service** | Subscribes to the fused state and renders a live 3-D matplotlib plot of the trajectory trail. |

---

## What the EKF Actually Does

The Extended Kalman Filter maintains a **state vector** — the system's best current guess of where the target is and how it is moving:

| State variables | Meaning |
|---|---|
| Position (x, y, z) | Where the target is in 3-D space, in metres |
| Velocity (x, y, z) | How fast and in which direction, in m/s |
| Orientation quaternion (w, x, y, z) | Which way the platform is pointing |

The filter runs a two-step loop:

### Step 1 — Predict (driven by IMU, 100 Hz)

Every time an IMU packet arrives, the EKF **propagates** the state forward in time:

- It rotates the IMU's body-frame acceleration into the world frame using the current orientation estimate.
- It integrates acceleration → velocity, and velocity → position.
- It integrates angular rate to update the orientation quaternion.
- Critically, it also **grows the uncertainty** (covariance matrix) to reflect that IMU integration accumulates error over time.

This gives a fast, smooth estimate between radar updates, but one that drifts.

### Step 2 — Update (driven by radar, 10 Hz)

Every time a radar packet arrives, the EKF **corrects** the state:

- It computes what the radar *should* have measured given the current state estimate (the predicted measurement).
- It compares this to what the radar *actually* measured (the innovation).
- It calculates a **Kalman gain** — how much to trust the radar versus the current prediction — based on the relative uncertainties of each.
- It shifts the state estimate toward the radar measurement, weighted by the Kalman gain.
- It **shrinks the uncertainty** because a real observation has arrived.

The result: the IMU keeps the estimate smooth and fast between radar pings, and the radar prevents the IMU from drifting.

---

## The Backpropagation Problem

Real sensor networks are messy. A radar packet can arrive at the processing node *after* a later IMU packet has already been processed — because of network jitter, OS scheduling, or buffering delays.

If you simply process packets as they arrive, you corrupt the filter state: you are updating with an out-of-order measurement as if it arrived in sequence, which is mathematically wrong.

### How this system solves it

The **buffer_service** detects late arrivals and flags them with `is_late = True`.

When the **ekf_service** receives a late-flagged packet it performs **backpropagation**:

1. It finds the saved EKF state snapshot taken just *before* the late measurement's timestamp.
2. It restores the EKF to that earlier state.
3. It re-inserts the late measurement at the correct chronological position and replays all subsequent measurements from the internal history buffer.

This retroactively corrects the trajectory without requiring any external coordination — the fix is entirely internal to the EKF service.

---

## Why the Microservice Architecture?

Each sensor and processing stage runs as a **separate process** connected through a broker:

- **Independent scaling** — you can run multiple radar or IMU simulators, or swap in real hardware drivers, without touching any other service.
- **Fault isolation** — if the visualizer crashes, the EKF keeps running. If a sensor goes silent, only that sensor is affected.
- **Replaceability** — swap the simulated radar for a real radar driver by pointing the same topic at a new publisher. No other service changes.
- **Observable** — every inter-service message passes through the broker, making the entire data flow inspectable or recordable at any point.

---

## What the Output Tells You

After every radar update (10 Hz), the EKF publishes a fused state. The monitor prints:

```
━━━ EKF State #   42  t=1710000012.345 ━━━
  Position   x=   87.234 m   y=   57.891 m   z=   17.562 m
  Velocity   x=   -2.998 m/s y=    1.501 m/s z=   -0.499 m/s
  Attitude   roll=   0.12°  pitch=  -0.08°  yaw=   1.23°
  Cov trace  4.2137e-01
```

- **Position** — where the EKF believes the target is right now, in metres from the sensor origin.
- **Velocity** — how fast and in which direction the target is moving.
- **Attitude** — the orientation of the tracking platform as roll/pitch/yaw angles (converted from the internal quaternion).
- **Covariance trace** — the sum of all diagonal uncertainty terms. A large value means the filter is uncertain; a small value means it has converged. Watching this number fall over time tells you the filter is working.

The **visualizer** renders the position history as a 3-D trail so you can see the target's trajectory evolve in real time, with a velocity arrow showing the current direction of travel.

---

## In One Sentence

This system fuses a slow-but-absolute radar with a fast-but-drifting IMU through a mathematically principled Extended Kalman Filter, handles out-of-order sensor data automatically, and serves the resulting 3-D position and velocity estimate to any number of consumers over a message bus — all as independently deployable services.
