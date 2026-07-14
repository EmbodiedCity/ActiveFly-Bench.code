Header header

uint8 ACCEPT_IF_IDLE = 0
uint8 REJECT_IF_BUSY = 1

int16 drone_id
uint32 seq_id
string frame_id
geometry_msgs/Point position
float64 yaw
float64 pitch
float64 position_tolerance
float64 yaw_tolerance_deg
float64 pitch_tolerance_deg
float64 timeout_sec
uint8 busy_policy
