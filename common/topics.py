"""
ZeroMQ topic name constants shared across all services.
"""

TOPIC_IMU    = "imu"
TOPIC_RADAR  = "radar"
TOPIC_STATE  = "state"   # EKF fused state output (kept for compatibility)
TOPIC_EKF_STATE = TOPIC_STATE   # alias – preferred name going forward

# Buffer / backpropagation service broadcasts replayed entries on this topic
TOPIC_BUFFER = "buffer"
