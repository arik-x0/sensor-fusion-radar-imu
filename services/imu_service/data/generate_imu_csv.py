"""
generate_imu_csv.py
====================
Generates a sample IMU recording (CSV) for use with the file-replay
IMU service.  Run once from inside the `data/` directory:

    python generate_imu_csv.py

Output
------
imu_data.csv  — 30 seconds of IMU data at 100 Hz (3 001 rows, incl. header)

Columns
-------
timestamp   UNIX epoch (float, seconds)
accel_x     m/s²  (body-frame X)
accel_y     m/s²  (body-frame Y)
accel_z     m/s²  (body-frame Z)
gyro_x      rad/s
gyro_y      rad/s
gyro_z      rad/s

The simulated body has a slow constant acceleration with mild rotation,
plus realistic Gaussian sensor noise.
"""

import csv
import math
import pathlib
import numpy as np

# ── parameters ──────────────────────────────────────────────────────────────
DURATION_S   = 30          # seconds of data to generate
HZ           = 100         # samples per second
START_TIME   = 1741826400.0  # Arbitrary epoch anchor (2026-03-13 00:00:00 UTC)

# True body dynamics (constant for this simple dataset)
# Gravity component along Z in body frame (approximate level flight)
GRAVITY_Z    = 9.81        # m/s²

# True specific force (what accelerometer measures = accel - gravity)
TRUE_ACCEL   = np.array([0.30,  0.10, -0.05])  # m/s² (without gravity bias)

# True angular rate (slow, gentle)
TRUE_GYRO    = np.array([0.020, -0.010, 0.005])  # rad/s

# Sensor noise std-devs (1-σ)
NOISE_ACCEL  = 0.05   # m/s²
NOISE_GYRO   = 0.005  # rad/s

# Random seed for reproducibility
SEED = 42

# ── generation ───────────────────────────────────────────────────────────────

def generate(output_path: pathlib.Path) -> None:
    rng = np.random.default_rng(SEED)
    dt  = 1.0 / HZ
    n   = int(DURATION_S * HZ) + 1   # inclusive of t=0 and t=DURATION_S

    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["timestamp", "accel_x", "accel_y", "accel_z",
             "gyro_x",   "gyro_y",   "gyro_z"]
        )

        for i in range(n):
            t = START_TIME + i * dt

            # Add a slowly varying sinusoidal component on top of the constant
            # acceleration to make the dataset more interesting.
            sin_mod = math.sin(2 * math.pi * 0.05 * i * dt)  # 0.05 Hz modulation

            accel = TRUE_ACCEL + np.array([0.02 * sin_mod, 0.01 * sin_mod, 0.0])
            accel += rng.normal(0.0, NOISE_ACCEL, 3)

            gyro  = TRUE_GYRO + np.array([0.001 * sin_mod, 0.0, 0.0])
            gyro  += rng.normal(0.0, NOISE_GYRO, 3)

            writer.writerow([
                f"{t:.6f}",
                f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
                f"{gyro[0]:.6f}",  f"{gyro[1]:.6f}",  f"{gyro[2]:.6f}",
            ])

    print(f"Wrote {n} rows → {output_path}")


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    generate(here / "imu_data.csv")
