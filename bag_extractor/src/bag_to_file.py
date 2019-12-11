#!/usr/bin/python

# Start up ROS pieces.
PKG = 'bag_extractor'
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
import tf
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from object_pose_interface_msgs.msg import SemanticMapObjectArray
from cv_bridge import CvBridge, CvBridgeError

import os
import sys

# Make sure to first run:
# rosrun image_transport republish compressed in:=/zed/zed_node/rgb/image_rect_color raw out:=/image_republished

class ImageCreator():
    def __init__(self):
        rospy.init_node(PKG)

        self.save_dir = rospy.get_param('save_dir', '/home/kodlab/saved_images/')

        self.bridge = CvBridge()

        self.sem_index = 0

        self.map_frame = "map"
        self.robot_frame = "turtlebot"
        self.camera_base_frame = "base_link"
        self.camera_optical_frame = "zed_left_camera_optical_frame"
        self.object_frame = "object0"

        self.tf_listener = tf.TransformListener()

        self.sub_image = rospy.Subscriber("/image_republished", Image, self.image_callback)
        self.sub_semantic = rospy.Subscriber("/pose_tracking/semantic_map_republished", SemanticMapObjectArray, self.semantic_callback)

        rospy.spin()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        image_time = data.header.stamp
        timestr = "%.6f" % image_time.to_sec()
        image_name = str(self.save_dir)+timestr+".jpg"
        cv2.imwrite(image_name, cv_image)
        self.tf_listener.waitForTransform(self.robot_frame, self.map_frame, image_time, rospy.Duration.from_sec(1))
        (trans_map_in_robot, rot_map_in_robot) = self.tf_listener.lookupTransform(self.robot_frame, self.map_frame, image_time)
        rot_map_in_robot_euler = tf.transformations.euler_from_quaternion(rot_map_in_robot)
        map_in_robot = tf.transformations.compose_matrix(None, None, rot_map_in_robot_euler, trans_map_in_robot, None)

        if (self.sem_index != 0):
            object_in_camoptical = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(self.cambase_in_camoptical, self.robot_in_cambase), map_in_robot), self.robot_in_map_0), self.cambase_in_robot), self.camoptical_in_cambase), self.object_in_camoptical_0)
            object_in_camoptical /= object_in_camoptical[-1,-1]
            trans_object_in_camoptical = object_in_camoptical[0:3,-1]
            r, p, y = tf.transformations.euler_from_matrix(object_in_camoptical)
            rot_object_in_camoptical = tf.transformations.quaternion_from_euler(r,p,y)
            # self.tf_listener.waitForTransform(self.map_frame, self.robot_frame, image_time, rospy.Duration.from_sec(1))
            # (trans_robot_in_map, rot_robot_in_map) = self.tf_listener.lookupTransform(self.map_frame, self.robot_frame, image_time)
            np.savez(str(self.save_dir)+timestr+".npz", trans=np.array([trans_object_in_camoptical[0], trans_object_in_camoptical[1], trans_object_in_camoptical[2]]), quat=np.array([rot_object_in_camoptical[0], rot_object_in_camoptical[1], rot_object_in_camoptical[2], rot_object_in_camoptical[3]]))

        return

    def semantic_callback(self, data):
        if (self.sem_index == 0):
            trans_cambase_in_robot_x = 0.08
            trans_cambase_in_robot_y = -0.03
            trans_cambase_in_robot_z = 0

            self.tf_listener.waitForTransform(self.camera_optical_frame, self.camera_base_frame, rospy.Time(0), rospy.Duration.from_sec(10))
            print("Found transform of camera base frame in camera optical frame")
            (self.trans_cambase_in_camoptical, self.rot_cambase_in_camoptical) = self.tf_listener.lookupTransform(self.camera_optical_frame, self.camera_base_frame, rospy.Time(0))
            self.rot_cambase_in_camoptical_euler = tf.transformations.euler_from_quaternion(self.rot_cambase_in_camoptical)
            self.cambase_in_camoptical = tf.transformations.compose_matrix(None, None, self.rot_cambase_in_camoptical_euler, self.trans_cambase_in_camoptical, None)

            self.tf_listener.waitForTransform(self.camera_base_frame, self.robot_frame, rospy.Time(0), rospy.Duration.from_sec(10))
            print("Found transform of robot frame in camera base frame")
            (self.trans_robot_in_cambase, self.rot_robot_in_cambase) = self.tf_listener.lookupTransform(self.camera_base_frame, self.robot_frame, rospy.Time(0))
            self.trans_robot_in_cambase[0] = -trans_cambase_in_robot_x
            self.trans_robot_in_cambase[1] = -trans_cambase_in_robot_y
            self.trans_robot_in_cambase[2] = -trans_cambase_in_robot_z
            self.rot_robot_in_cambase_euler = tf.transformations.euler_from_quaternion(self.rot_robot_in_cambase)
            self.robot_in_cambase = tf.transformations.compose_matrix(None, None, self.rot_robot_in_cambase_euler, self.trans_robot_in_cambase, None)

            self.tf_listener.waitForTransform(self.map_frame, self.robot_frame, rospy.Time(0), rospy.Duration.from_sec(10))
            print("Found transform of robot frame in map frame for time 0")
            (self.trans_robot_in_map_0, self.rot_robot_in_map_0) = self.tf_listener.lookupTransform(self.map_frame, self.robot_frame, rospy.Time(0))
            self.rot_robot_in_map_0_euler = tf.transformations.euler_from_quaternion(self.rot_robot_in_map_0)
            self.robot_in_map_0 = tf.transformations.compose_matrix(None, None, self.rot_robot_in_map_0_euler, self.trans_robot_in_map_0, None)

            self.tf_listener.waitForTransform(self.robot_frame, self.camera_base_frame, rospy.Time(0), rospy.Duration.from_sec(10))
            print("Found transform of camera base frame in robot frame")
            (self.trans_cambase_in_robot, self.rot_cambase_in_robot) = self.tf_listener.lookupTransform(self.robot_frame, self.camera_base_frame, rospy.Time(0))
            self.trans_cambase_in_robot[0] = trans_cambase_in_robot_x
            self.trans_cambase_in_robot[1] = trans_cambase_in_robot_y
            self.trans_cambase_in_robot[2] = trans_cambase_in_robot_z
            self.rot_cambase_in_robot_euler = tf.transformations.euler_from_quaternion(self.rot_cambase_in_robot)
            self.cambase_in_robot = tf.transformations.compose_matrix(None, None, self.rot_cambase_in_robot_euler, self.trans_cambase_in_robot, None)

            self.tf_listener.waitForTransform(self.camera_base_frame, self.camera_optical_frame, rospy.Time(0), rospy.Duration.from_sec(10))
            print("Found transform of camera optical frame in camera base frame")
            (self.trans_camoptical_in_cambase, self.rot_camoptical_in_cambase) = self.tf_listener.lookupTransform(self.camera_base_frame, self.camera_optical_frame, rospy.Time(0))
            self.rot_camoptical_in_cambase_euler = tf.transformations.euler_from_quaternion(self.rot_camoptical_in_cambase)
            self.camoptical_in_cambase = tf.transformations.compose_matrix(None, None, self.rot_camoptical_in_cambase_euler, self.trans_camoptical_in_cambase, None)

            self.tf_listener.waitForTransform(self.camera_optical_frame, self.object_frame, rospy.Time(0), rospy.Duration.from_sec(10))
            print("Found transform of object frame in camera optical frame for time 0")
            (self.trans_object_in_camoptical_0, self.rot_object_in_camoptical_0) = self.tf_listener.lookupTransform(self.camera_optical_frame, self.object_frame, rospy.Time(0))
            self.rot_object_in_camoptical_0_euler = tf.transformations.euler_from_quaternion(self.rot_object_in_camoptical_0)
            self.object_in_camoptical_0 = tf.transformations.compose_matrix(None, None, self.rot_object_in_camoptical_0_euler, self.trans_object_in_camoptical_0, None)

            self.sem_index = 1

        return

# Main function.    
if __name__ == '__main__':
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException: pass
