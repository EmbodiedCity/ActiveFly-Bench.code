# Multi-waypoint VLA command for trajectory-level action chunk execution.
Header header

uint8 ACCEPT_IF_IDLE = 0
uint8 REJECT_IF_BUSY = 1
uint8 REPLACE_ACTIVE = 2

int16 drone_id
uint32 seq_id
string frame_id
geometry_msgs/Point[] positions
float64[] yaws
float64[] pitches
float64 position_tolerance
float64 yaw_tolerance_deg
float64 pitch_tolerance_deg
float64 timeout_sec
uint8 busy_policy

uint32 query_id
int32 query_step
int32 first_chunk_index
