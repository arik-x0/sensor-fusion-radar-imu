"""
ZeroMQ topic name constants shared across all services.
"""

TOPIC_ROVER_TRUTH = "rover_truth"  # Ground-truth rover state (rover_service → radar/imu)
TOPIC_IMU         = "imu"
TOPIC_RADAR       = "radar"
TOPIC_STATE       = "state"        # EKF fused state output
TOPIC_EKF_STATE   = TOPIC_STATE    # alias – preferred name going forward

# Buffer / backpropagation service broadcasts replayed entries on this topic
TOPIC_BUFFER = "buffer"
