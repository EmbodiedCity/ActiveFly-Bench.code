# gimbal

Unified ROS package for gimbal camera and serial bridge control.

## Layout

- `src/camera`: camera capture and image publish node
- `src/serial`: serial bridge node for gimbal command/state communication
- `scripts`: command-line teleop and monitoring tools
- `msg`: `GimbalCmd` and `GimbalState` message definitions
- `launch`: launch files for camera and serial bridge
- `config`: default runtime parameters
- `docs`: test and debugging notes

## Nodes

- `gimbal_camera_node`: publishes `/gimbal/image_raw` and `/gimbal/camera_info`
- `gimbal_ctrl`: subscribes `/gimbal/cmd` and publishes `/gimbal/state`

## CLI Teleop

Use the interactive CLI tool for real-time pitch-axis control and live status display:

```bash
rosrun gimbal gimbal_cli_teleop.py
```

Default topics:

- command publish: `/gimbal/cmd`
- state subscribe: `/gimbal/state`

Main keys:

- `W` or up arrow: continuous pitch positive direction
- `S` or down arrow: continuous pitch negative direction
- `C`: one-key center (pitch target set to 0 deg)
- `M`: send manual mode command (`cmd=4`)
- `T`: send start command (`cmd=2`)
- `X`: send stop command (`cmd=3`)
- `D`: increase pitch speed
- `A`: decrease pitch speed
- `Q`: quit (auto-center on exit by default)

Useful parameters:

```bash
rosrun gimbal gimbal_cli_teleop.py _pitch_min_deg:=-90 _pitch_max_deg:=40 _pitch_rate_deg_per_sec:=40 _center_on_exit:=true
```

## Increment CLI

Use the incremental terminal tool for a realtime fullscreen UI similar to the teleop tool:

```bash
rosrun gimbal gimbal_pitch_increment_cli.py
```

Behavior:

- realtime curses-style screen refresh with no scrolling log output
- type a numeric delta such as `-20` or `5.5`, then press `Enter`: add the increment to the last target pitch sent by this script
- first command fallback: if this script has not sent any target yet, it uses the latest `/gimbal/state` camera pitch as the base angle
- press `0` then `Enter`, or press `C`: one-key center (target pitch set to `0 deg`)
- press `[` / `]`: apply one negative / positive step using `_increment_step_deg`
- press `Q`: quit
- all target angles are clamped into configured pitch limits before publish
- all UI text is in English

Useful parameters:

```bash
rosrun gimbal gimbal_pitch_increment_cli.py _pitch_min_deg:=-90 _pitch_max_deg:=40 _state_timeout_sec:=1.5 _increment_step_deg:=5
```

Serial bridge control parameters (startup lock pose + realtime pitch control):

```bash
rosrun gimbal gimbal_ctrl _serial_path:=/dev/ttyUSB0 _locked_roll_deg:=0.0 _locked_pitch_deg:=0.0 _locked_yaw_deg:=90.0 _pitch_ctrl_min_deg:=-90.0 _pitch_ctrl_max_deg:=40.0 _yaw_wk_mode:=1 _fpv_follow_sens:=15 _rx_first_timeout_ms:=4 _io_sleep_ms:=1 _check_mount_mode:=true _expected_inv_flag:=0
```

Axis stiffness tuning:

- `yaw_wk_mode`: yaw axis mode (`0=follow`, `1=lock`, `2=FPV`). Startup lock profile uses `1`.
- `locked_roll_deg`: roll lock target angle (default `0 deg`).
- `pitch_ctrl_min_deg` / `pitch_ctrl_max_deg`: realtime pitch command range.
- `fpv_follow_sens`: FPV follow sensitivity (`-16..15`), only meaningful when yaw in FPV mode.
- `rx_first_timeout_ms`: first-byte read timeout for each cycle (`1..50`). Smaller value lowers waiting latency.
- `io_sleep_ms`: loop sleep after each cycle (`0..20`). Smaller value improves update cadence at higher CPU load.

Recommended rigid profile:

```bash
roslaunch gimbal gimbal_serial_bridge.launch serial_path:=/dev/USB1 locked_roll_deg:=0 locked_yaw_deg:=90 pitch_ctrl_min_deg:=-90 pitch_ctrl_max_deg:=40 yaw_wk_mode:=1 rx_first_timeout_ms:=4 io_sleep_ms:=1
```

Mount mode check:

- `expected_inv_flag=0`: hanging/inverted
- `expected_inv_flag=1`: upright
