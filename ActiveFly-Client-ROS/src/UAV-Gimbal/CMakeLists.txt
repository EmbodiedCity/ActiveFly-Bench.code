cmake_minimum_required(VERSION 3.0.2)
project(gimbal)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  camera_info_manager
  cv_bridge
  geometry_msgs
  image_transport
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_message_files(
  FILES
  GimbalCmd.msg
  GimbalState.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS camera_info_manager cv_bridge geometry_msgs image_transport message_runtime roscpp rospy sensor_msgs std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(gimbal_camera_node
  src/camera/gimbal_camera_node.cpp
)

target_link_libraries(gimbal_camera_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

add_executable(gimbal_ctrl
  src/serial/gimbal_ctrl.cpp
)

add_dependencies(gimbal_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(gimbal_ctrl
  ${catkin_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS gimbal_camera_node gimbal_ctrl
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/gimbal_cli_teleop.py
  scripts/gimbal_pitch_increment_cli.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
