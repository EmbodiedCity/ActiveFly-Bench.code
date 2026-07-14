Header header

uint8 IDLE = 0
uint8 BUSY = 1
uint8 DONE = 2
uint8 REJECTED_BUSY = 3
uint8 PLAN_FAILED = 4
uint8 HOLD_FAILED = 5
uint8 GIMBAL_TIMEOUT = 6
uint8 COMMAND_TIMEOUT = 7
uint8 ABORTED = 8

int16 drone_id
uint32 seq_id
uint8 status
string message
geometry_msgs/Point target_position
float64 target_yaw
float64 target_pitch
float64 position_error
float64 yaw_error_deg
float64 pitch_error_deg
