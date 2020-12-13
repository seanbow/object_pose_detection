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
from visualization_msgs.msg import MarkerArray

import os
import sys


class MarkerRepublisher():
	def __init__(self):
		rospy.init_node('marker_republisher')

		self.pub_markers = rospy.Publisher("/object_markers_republished", MarkerArray, queue_size=1)
		self.sub_markers = rospy.Subscriber("/pose_tracking/keypoint_objects/object_markers", MarkerArray, self.marker_callback)

		rospy.spin()

	def marker_callback(self, data):
		for i in range(0,len(data.markers)):
			if data.markers[i].text == 'cart':
				data.markers[i].scale.x = 0.08
				data.markers[i].scale.y = 0.08
				data.markers[i].scale.z = 0.08
		
		self.pub_markers.publish(data)

		return
    
# Main function.    
if __name__ == '__main__':
    try:
        marker_republisher = MarkerRepublisher()
    except rospy.ROSInterruptException: pass
