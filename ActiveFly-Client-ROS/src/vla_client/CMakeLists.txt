cmake_minimum_required(VERSION 3.0.2)
project(vla_client)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
  VlaStatus.msg
  VlaWaypoint.msg
  VlaWaypointQueue.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    geometry_msgs
    message_runtime
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
)

catkin_install_python(PROGRAMS
  scripts/openvla_client.py
  scripts/openvla_chunking_client.py
  scripts/pi05_chunking_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
