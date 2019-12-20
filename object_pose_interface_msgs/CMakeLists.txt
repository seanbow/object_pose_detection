cmake_minimum_required(VERSION 2.8.3)
project(object_pose_interface_msgs)

find_package(catkin REQUIRED 
  COMPONENTS
  roscpp
  message_generation
  std_msgs
  geometry_msgs
  darknet_ros_msgs
)

add_message_files(
  FILES
  Classification.msg
  DetectedObject.msg
  DetectedObjectArray.msg
  ira_det.msg
  ira_dets.msg
  KeypointDetection.msg
  KeypointDetection3D.msg
  KeypointDetections.msg
  KeypointDetections3D.msg
  ObjectClass.msg
  ObjectClassArray.msg
  SemanticMapObject.msg
  SemanticMapObjectArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  darknet_ros_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_pose_interface_msgs
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
#  DEPENDS system_lib
)
