#!/usr/bin/env python

"""
MIT License (modified)

Copyright (c) 2019 The Trustees of the University of Pennsylvania
Authors:
Vasileios Vasilopoulos <vvasilo@seas.upenn.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this **file** (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import rospy, struct, math, tf, os, sys
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes
import tf


# Publisher
global pub


def read_detection(data):
	"""
	Function that reads the detection and publishes it back with a modified header
	"""
	# Publisher
	global pub

	# Read data and publish
	data.header = data.image_header
	pub.publish(data)

	return
	

def init():
	# Publisher
	global pub
	
	# Initialize node
	rospy.init_node('detection_republisher', anonymous = True)

	# Find parameters
	detection_topic = rospy.get_param('~detection_topic')
	detection_republished_topic = rospy.get_param('~detection_republished_topic')

	# Define publisher
	pub = rospy.Publisher(detection_republished_topic, BoundingBoxes, queue_size=1)

	# Define subscriber
	rospy.Subscriber(detection_topic, BoundingBoxes, read_detection)
	
	# Spin
	rospy.spin()

if __name__ == '__main__':
	try:
		init()
	except rospy.ROSInterruptException: pass
