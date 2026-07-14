
uint8 inv_fag  # 吊装/立装标志 0- 吊装；1- 立装
uint8 gbc_stat # 云台状态 0- 未定义；1- 初始化中；2- 云台停止；3- 云台保护；4- 手动控制；5- 指点平移
uint8 tca_fag  # 温控就绪标志 0- 未就绪
uint8 cmd_stat # 命令执行状态 0- 执行中；1- 执行成功；2- 执行失败
uint8 cmd_value #命令码回报

# 相机姿态角速度
int16 cam_rate_p #俯仰
int16 cam_rate_r #滚转
int16 cam_rate_y #偏航

# 相机姿态角
int16 cam_angle_p #俯仰
int16 cam_angle_r #滚转
int16 cam_angle_y #偏航

# 电机相对角
int16 mtr_angle_p #俯仰
int16 mtr_angle_r #滚转
int16 mtr_angle_y #偏航

