# ActiveFly VLA Client ROS Workspace

This repository is a minimal ROS1 catkin workspace for real-UAV VLA client deployment. It contains the ROS-side clients for OpenVLA and OpenPi pi0.5, the shared VLA command/status messages, and the gimbal driver package used by the deployment.

It does not include model weights, OpenVLA/OpenPi inference servers, planner or controller nodes.

## Repository Layout

```text
opensource_vla_client_ws/
  openpi-client/             # Lightweight Python client used by the pi0.5 ROS node
  src/
    vla_client/              # VLA ROS messages and OpenVLA/OpenPi client nodes
      msg/
        VlaWaypoint.msg
        VlaWaypointQueue.msg
        VlaStatus.msg
      scripts/
        openvla_client.py
        openvla_chunking_client.py
        pi05_chunking_client.py
    UAV-Gimbal/              # Gimbal driver, messages, launch files, and configs
```

## Build

Install ROS1 and the system dependencies required by your ROS distribution. The commands below assume Ubuntu with ROS Noetic:

```bash
cd opensource_vla_client_ws
catkin_make
source devel/setup.bash
```

Python dependencies:

```bash
python3 -m pip install numpy pillow requests
python3 -m pip install -e ./openpi-client
```

The `openpi-client` package adds the WebSocket policy client used by `pi05_chunking_client.py`.

## Run

Start the gimbal driver:

```bash
source devel/setup.bash
roslaunch gimbal start_gimbal.launch
```

Run OpenVLA chunking client:

```bash
rosrun vla_client openvla_chunking_client.py \
  --server_url http://<server_ip>:5007/predict \
  --instruction "Describe the UAV task here."
```

Run OpenPi pi0.5 chunking client:

```bash
rosrun vla_client pi05_chunking_client.py \
  --server_ip <server_ip> \
  --server_port 8000 \
  --instruction "Describe the UAV task here."
```

For first real-flight tests, keep the default keyboard confirmation behavior. Add `--no_user_confirm` only after verifying the generated waypoints, planner feedback, and gimbal motion are correct.

Runtime artifacts are written under `~/.ros/vla_client_runs` by default. Use `--output_dir <path>` to change that location.

## ROS Interfaces

The client package provides these messages:

- `vla_client/VlaWaypoint`
- `vla_client/VlaWaypointQueue`
- `vla_client/VlaStatus`

Default subscriptions:

- `/ekf_quat/ekf_odom`
- `/gimbal/image_raw`
- `/gimbal/state`
- `/vla_status`

Default publications:

- `/vla_waypoint`
- `/vla_waypoint_queue`
- `/predicted_world_pose`

If your planner still uses `quadrotor_msgs/VlaWaypoint`, `quadrotor_msgs/VlaWaypointQueue`, or `quadrotor_msgs/VlaStatus`, update the planner message imports to `vla_client/*` or add a small bridge node between the two message namespaces.

## Safety Notes

These scripts publish real motion targets. Before flight, verify odometry, gimbal state, image topic, VLA server reachability, planner status feedback, RC/manual takeover, and emergency stop behavior. Start with conservative tolerances and low speed, and inspect `/vla_waypoint` and `/vla_status` before enabling unattended execution.
