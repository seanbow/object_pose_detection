#!/usr/bin/python

# Start up ROS pieces.
PKG = 'bag_extractor'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
import tf
import numpy as np
import scipy.io as sio
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from object_pose_interface_msgs.msg import SemanticMapObjectArray
from cv_bridge import CvBridge, CvBridgeError

import os
import sys


class PathPublisher():
	def __init__(self):
		rospy.init_node('path_publisher')

		self.path = Path()
		self.flag = True
		self.offset_x = 0.0
		self.offset_y = 0.0
		self.offset_z = 0.0

		self.pub_path = rospy.Publisher("/path", Path, queue_size=1)
		self.pub_goal = rospy.Publisher("/goal_point", PointStamped, queue_size=1)
		self.sub_odom_vicon = rospy.Subscriber("/vicon/turtlebot/odom", Odometry, self.vicon_odom)

		rospy.spin()

	def vicon_odom(self, data):
		self.path.header = data.header
		self.path.header.frame_id = 'map'
	
		if self.flag:
			self.offset_x = data.pose.pose.position.x
			self.offset_y = data.pose.pose.position.y
			self.offset_z = data.pose.pose.position.z
			self.flag = False

		pose_to_add = PoseStamped()
		pose_to_add.header = data.header
		pose_to_add.pose.position.x = data.pose.pose.position.x-self.offset_x
		pose_to_add.pose.position.y = data.pose.pose.position.y-self.offset_y
		pose_to_add.pose.position.z = data.pose.pose.position.z-self.offset_z
		pose_to_add.pose.orientation.x = data.pose.pose.orientation.x
		pose_to_add.pose.orientation.y = data.pose.pose.orientation.y
		pose_to_add.pose.orientation.z = data.pose.pose.orientation.z
		pose_to_add.pose.orientation.w = data.pose.pose.orientation.w

		self.path.poses.append(pose_to_add)
		self.pub_path.publish(self.path)

		goal_point = PointStamped()
		goal_point.header.stamp = data.header.stamp
		goal_point.header.frame_id = 'map'
		goal_point.point.x = 0.0
		goal_point.point.y = 0.0
		self.pub_goal.publish(goal_point)

		return
    
# Main function.    
if __name__ == '__main__':
    try:
        path_publisher = PathPublisher()
    except rospy.ROSInterruptException: pass
