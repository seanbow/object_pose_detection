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
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from object_pose_interface_msgs.msg import SemanticMapObjectArray
from cv_bridge import CvBridge, CvBridgeError

import os
import sys


class OdometryExtractor():
    def __init__(self):
        rospy.init_node(PKG)

        self.save_dir = rospy.get_param('save_dir', '/home/kodlab/experiments/')

        self.first_index = 0

        self.raw_odom_translation = []
        self.raw_odom_rotation = []

        self.semslam_odom_translation = []
        self.semslam_odom_rotation = []

        self.vicon_odom_translation = []
        self.vicon_odom_rotation = []

        self.sub_odom_raw = rospy.Subscriber("/pose_tracking/stereo_camera_odom_republished", Odometry, self.raw_odom)
        self.sub_odom_semslam = rospy.Subscriber("/pose_tracking/semslam/robot_odom_republished", Odometry, self.semslam_odom)
        self.sub_odom_vicon = rospy.Subscriber("/vicon/turtlebot/odom", Odometry, self.vicon_odom)

        rospy.spin()
    
    def raw_odom(self,data):
        position_x = data.pose.pose.position.x
        position_y = data.pose.pose.position.y
        position_z = data.pose.pose.position.z
        orientation_x = data.pose.pose.orientation.x
        orientation_y = data.pose.pose.orientation.y
        orientation_z = data.pose.pose.orientation.z
        orientation_w = data.pose.pose.orientation.w

        translation = np.array([[position_x], [position_y], [position_z]]).transpose()
        rotation = tf.transformations.quaternion_matrix([orientation_x, orientation_y, orientation_z, orientation_w])

        self.raw_odom_translation.append(translation)
        self.raw_odom_rotation.append(rotation[0:3,0:3])
        sio.savemat(self.save_dir + 'raw_odom.mat', mdict={'translation': np.array(self.raw_odom_translation), 'rotation': np.array(self.raw_odom_rotation)})

        return
    
    def semslam_odom(self,data):
        position_x = data.pose.pose.position.x
        position_y = data.pose.pose.position.y
        position_z = data.pose.pose.position.z
        orientation_x = data.pose.pose.orientation.x
        orientation_y = data.pose.pose.orientation.y
        orientation_z = data.pose.pose.orientation.z
        orientation_w = data.pose.pose.orientation.w

        translation = np.array([[position_x], [position_y], [position_z]]).transpose()
        rotation = tf.transformations.quaternion_matrix([orientation_x, orientation_y, orientation_z, orientation_w])

        self.semslam_odom_translation.append(translation)
        self.semslam_odom_rotation.append(rotation[0:3,0:3])
        sio.savemat(self.save_dir + 'semslam_odom.mat', mdict={'translation': np.array(self.semslam_odom_translation), 'rotation': np.array(self.semslam_odom_rotation)})

        return

    def vicon_odom(self, data):
        if (self.first_index == 0):
            self.first_position_x = data.pose.pose.position.x
            self.first_position_y = data.pose.pose.position.y
            self.first_position_z = data.pose.pose.position.z
            self.first_orientation_x = data.pose.pose.orientation.x
            self.first_orientation_y = data.pose.pose.orientation.y
            self.first_orientation_z = data.pose.pose.orientation.z
            self.first_orientation_w = data.pose.pose.orientation.w

            self.first_translation = np.array([[self.first_position_x], [self.first_position_y], [self.first_position_z]])
            self.first_rotation = tf.transformations.quaternion_matrix([self.first_orientation_x, self.first_orientation_y, self.first_orientation_z, self.first_orientation_w])

            self.first_index = 1
        
        position_x = data.pose.pose.position.x
        position_y = data.pose.pose.position.y
        position_z = data.pose.pose.position.z
        orientation_x = data.pose.pose.orientation.x
        orientation_y = data.pose.pose.orientation.y
        orientation_z = data.pose.pose.orientation.z
        orientation_w = data.pose.pose.orientation.w

        translation = np.array([[position_x], [position_y], [position_z]])
        rotation = tf.transformations.quaternion_matrix([orientation_x, orientation_y, orientation_z, orientation_w])
        translation = np.matmul(self.first_rotation[0:3,0:3].transpose(),translation-self.first_translation)
        rotation = np.matmul(self.first_rotation[0:3,0:3].transpose(),rotation[0:3,0:3])
        

        self.vicon_odom_translation.append(translation.transpose())
        self.vicon_odom_rotation.append(rotation)
        sio.savemat(self.save_dir + 'vicon_odom.mat', mdict={'translation': np.array(self.vicon_odom_translation), 'rotation': np.array(self.vicon_odom_rotation)})

        return

# Main function.    
if __name__ == '__main__':
    try:
        odometry_extractor = OdometryExtractor()
    except rospy.ROSInterruptException: pass
