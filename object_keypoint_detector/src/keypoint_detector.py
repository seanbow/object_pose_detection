#!/usr/bin/env python
import rospy
import rospkg

import time
from math import floor, ceil, sqrt
import cv2
import numpy as np
from os.path import join
import torch

import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from object_pose_interface_msgs.msg import ira_dets
from object_pose_interface_msgs.msg import KeypointDetection
from object_pose_interface_msgs.msg import KeypointDetections

from models import SmallStackedHourglass
from models import LargeStackedHourglass

class KeypointDetectorNode(object):
    ''' This node uses a stacked hourglass network implemented in pytorch to
    find the locations of keypoints in an image.
    '''
    def __init__(self):
        rospy.init_node('keypoint_detector_node')
        self.heatmap_pub = rospy.Publisher('pose_estimator/heatmap', Image,
                                           queue_size=1)
        self.heatmap_test_pub = rospy.Publisher('pose_estimator/heatmap_test',
                                                Image, queue_size=1)
        self.keypoints_pub = rospy.Publisher('pose_estimator/keypoints',
                                             KeypointDetections, queue_size=5)

        self.img_keypoints_pub = rospy.Publisher('pose_estimator/img_keypoints',
                                             KeypointDetections, queue_size=50)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo("Using %s to run stacked hourglass", self.device)

        num_hourglasses = rospy.get_param('~num_hourglasses', 4)
        hourglass_channels = rospy.get_param('~hourglass_channels', 256)
        self.img_size = rospy.get_param('~img_size', 256)
        self.detection_thresh = rospy.get_param('~detection_threshold', 0.01)
        self.yolo = rospy.get_param('/yolo')
        self.activation_threshold = rospy.get_param('~keypoint_activation_threshold')
        
        
        if rospy.has_param('~image_topic'):
            image_topic_set = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
            self.image_topic = image_topic_set
        else:
            rospy.logerr("No image topic specified")
        
        self.width = 0
        self.height = 0

        rospack = rospkg.RosPack()
        model_base_path = join(rospack.get_path('object_keypoint_detector'),
                               'models',
                               'keypoint_localization')

        num_keypoints_file_path = rospy.get_param('~num_keypoints_file',
                                                  'num_keypoints.txt')
        num_keypoints_file_path = join(model_base_path, num_keypoints_file_path)
        model_path = rospy.get_param('~model_path', "keypoints.pt")
        model_path = join(model_base_path, model_path)

        self.keypoints_indices = dict()
        start_index = 0
        
        with open(num_keypoints_file_path, 'r') as num_keypoints_file:
            for line in num_keypoints_file:
                split = line.split(' ')
                if len(split) == 2:
                    self.keypoints_indices[split[0]] = \
                        (start_index, start_index + int(split[1]))
                    start_index += int(split[1])
        print "Keypoint indices:", self.keypoints_indices


        model_type = rospy.get_param('~model_type', 'small')
        if model_type == 'small':
            self.model = SmallStackedHourglass(
                num_hg=num_hourglasses,
                hg_channels=hourglass_channels,
                out_channels=start_index)
        elif model_type == 'large':
            self.model = LargeStackedHourglass(out_channels=start_index)
        else:
            rospy.logerr("%s is an invalid model type", model_type)
            return
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['stacked_hg'])
        self.model.eval()
        rospy.loginfo("Loaded model")
        self.bridge = CvBridge()

        # detections_sub = message_filters.Subscriber('detected_objects',
                                                    # ira_dets)
        
        if (self.yolo):
            detections_sub = message_filters.Subscriber('detected_objects', BoundingBoxes)
        else:
            detections_sub = message_filters.Subscriber('detected_objects', ira_dets)

        image_sub = message_filters.Subscriber(self.image_topic, Image)
        combined_sub = message_filters.TimeSynchronizer(
            [detections_sub, image_sub], 100)
        combined_sub.registerCallback(self.detect_keypoints)
        rospy.loginfo("Spinning")
        rospy.spin()

    def detect_keypoints(self, detections_msg, image_msg):
        if (self.yolo):
            rospy.loginfo("Got %d detections in a %d x %d image", len(detections_msg.bounding_boxes), image_msg.width, image_msg.height)
        else:
            rospy.loginfo("Got %d detections in a %d x %d image", len(detections_msg.dets), image_msg.width, image_msg.height)


        if (self.width==0) and (self.height==0):
            self.width = image_msg.width
            self.height = image_msg.height
            preallocated_img_size = [2*(self.width+self.height), 2*(self.width+self.height), 3]
            self.img_preallocated = np.zeros(preallocated_img_size).astype(np.float32)
            original_img_size = [self.height, self.width, 3]
            self.img = np.zeros(original_img_size).astype(np.float32)
            self.img_published = np.zeros(original_img_size).astype(np.uint8)


        before_time = time.clock()
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        except CvBridgeError as error:
            rospy.logerr(error)
            return
        self.img = image.astype(np.float32)/255.0
        self.img_preallocated[self.width+self.height//2:self.width+self.height//2+self.height,self.height+self.width//2:self.height+self.width//2+self.width,:] = self.img
        self.img_published = image.astype(np.uint8)

        keypoint_detections = KeypointDetections()
        keypoint_detections.header = detections_msg.header
        
        img_keypoint_detections = KeypointDetections()
        img_keypoint_detections.header = detections_msg.header

        if (self.yolo):
            if not detections_msg.bounding_boxes:
                self.keypoints_pub.publish(keypoint_detections)
                self.img_keypoints_pub.publish(img_keypoint_detections)
                
                image_msg = self.bridge.cv2_to_imgmsg(self.img_published, encoding="rgb8")
                self.heatmap_pub.publish(image_msg)
                return
        else:
            if not detections_msg.dets:
                self.keypoints_pub.publish(keypoint_detections)
                self.img_keypoints_pub.publish(img_keypoint_detections)
                
                image_msg = self.bridge.cv2_to_imgmsg(self.img_published, encoding="rgb8")
                self.heatmap_pub.publish(image_msg)
                return
        
        if (self.yolo):
            bounding_boxes_detected = detections_msg.bounding_boxes
        else:
            bounding_boxes_detected = detections_msg.dets
        
        patches, bounds = self.get_patches_and_bounds(bounding_boxes_detected, self.img_preallocated)

        pred_keypoints = self.get_keypoints(patches)

        for i, detection in enumerate(bounding_boxes_detected):
            detection_msg = KeypointDetection()
            obj_name = (detection.Class) if (self.yolo) else (detection.obj_name)
            detection_msg.obj_name = obj_name
            detection_msg.header = detections_msg.header
            detection_msg.bounding_box = detection
            predictions = pred_keypoints[i, :, :, :]
            
            img_detection_msg = KeypointDetection()
            
            img_detection_msg.header = detections_msg.header
            img_detection_msg.obj_name = obj_name
            img_detection_msg.bounding_box = detection
            
            if obj_name not in self.keypoints_indices:
                continue
            
            for j in range(self.keypoints_indices[obj_name][0], self.keypoints_indices[obj_name][1]):
                coords = np.unravel_index(np.argmax(predictions[j]), predictions[j, :, :].shape)
                detection_msg.x.append(coords[1])
                detection_msg.y.append(coords[0])
                detection_msg.probabilities.append(predictions[j, coords[0], coords[1]])
                
                img_coords = [0, 0]
                img_coords[0] = bounds[i][0] + int(1.0 * coords[1] / predictions.shape[-2] * (bounds[i][1] - bounds[i][0]) + 0.5)
                img_coords[1] = bounds[i][2] + int(1.0 * coords[0] / predictions.shape[-1] * (bounds[i][3] - bounds[i][2]) + 0.5)
                
                img_detection_msg.x.append(img_coords[0])
                img_detection_msg.y.append(img_coords[1])
                img_detection_msg.probabilities.append(predictions[j, coords[0], coords[1]])
                
                # Red if the keypoint is less than our threshold, green otherwise
                if predictions[j, coords[0], coords[1]] < self.activation_threshold:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                    
                cv2.circle(self.img_published, (img_coords[0], img_coords[1]), 5, color, thickness=-1)
                
            keypoint_detections.detections.append(detection_msg)
            img_keypoint_detections.detections.append(img_detection_msg)

        self.keypoints_pub.publish(keypoint_detections)
        self.img_keypoints_pub.publish(img_keypoint_detections)

        image_msg = self.bridge.cv2_to_imgmsg(self.img_published, encoding="rgb8")
        self.heatmap_pub.publish(image_msg)

        rospy.loginfo("Found keypoints for %d objects in %f seconds", len(bounding_boxes_detected), time.clock() - before_time)


    def get_transform(self, center, scale, res, rot=0):
        """
        General image processing functions
        """
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot # To match direction of rotation from cropping
            rot_mat = np.zeros((3,3))
            rot_rad = rot * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
            rot_mat[2,2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0,2] = -res[1]/2
            t_mat[1,2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2,2] *= -1
            t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
        return t

    def transform(self, pt, center, scale, res, invert=0, rot=0):
        # Transform pixel location to different reference
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int) + 1
        
    def get_patches_and_bounds(self, detections, image):
        ''' Uses the detections to get the patches of the image that contain
        the detected objects.
        '''
        patches = np.zeros((len(detections), 3, self.img_size, self.img_size))
        bounds = []

        for i, detection in enumerate(detections):
            if (self.yolo):
                x_min = int(floor(detection.xmin))
                y_min = int(floor(detection.ymin))
                x_max = int(ceil(detection.xmax))
                y_max = int(ceil(detection.ymax))
            else:
                x_min = int(floor(detection.bbox.points[0].x))
                y_min = int(floor(detection.bbox.points[0].y))
                x_max = int(ceil(detection.bbox.points[2].x))
                y_max = int(ceil(detection.bbox.points[2].y))
            
            # factor to dilate the bbox
            scaleFactor = 1.2
            # width, height and center
            width = x_max - x_min
            height = y_max - y_min
            center = np.array([(x_min+x_max)/2., (y_min+y_max)/2.])
            # scale of dilated image
            scalePixels = scaleFactor*max(width, height)/2

            # Increases the size of the patch to ensure the entire object is included
            # We also shift everything to match the preallocated image
            x_min = int(center[0] - scalePixels + self.height + self.width/2)
            x_max = int(center[0] + scalePixels + self.height + self.width/2)
            y_min = int(center[1] - scalePixels + self.width + self.height/2)
            y_max = int(center[1] + scalePixels + self.width + self.height/2)
            
            # crop and resize
            patch = image[y_min:y_max, x_min:x_max, :]
            resized_patch = cv2.resize(patch, (self.img_size, self.img_size))
            
            # collect patches
            resized_patch = np.moveaxis(resized_patch, 2, 0)
            patches[i, :, : :] = resized_patch

            # Make sure to shift everything back to match the original image
            x_min = int(x_min - self.height - self.width/2)
            x_max = int(x_max - self.height - self.width/2)
            y_min = int(y_min - self.width - self.height/2)
            y_max = int(y_max - self.width - self.height/2)

            bounds.append([x_min, x_max, y_min, y_max])
        return patches, bounds

    def get_keypoints(self, patches):
        ''' Runs the images through the network
        '''
        patches_tensor = torch.from_numpy(patches)
        patches_tensor = patches_tensor.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            heatmaps = self.model(patches_tensor)
        # The network returns a list of the outputs of all of the hourglasses
        # in the stack.  We want the output of the final hourglass
        return heatmaps[-1].cpu().numpy()

    def generate_heatmap_grid(self, keypoints, object_type):
        ''' Generates a grid of heatmaps for a single detection
        '''
        num_images = self.keypoints_indices[object_type][1] - \
            self.keypoints_indices[object_type][0]

        grid_size = int(ceil(sqrt(num_images)))
        combined_keypoints = np.zeros((64*grid_size, 64*grid_size),
                                      dtype=np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j + self.keypoints_indices[object_type][0]
                if index >= self.keypoints_indices[object_type][1]:
                    continue
                print keypoints[index].shape
                combined_keypoints[i*64:(i+1)*64, j*64:(j+1)*64] = keypoints[index]
        return combined_keypoints

if __name__ == '__main__':
    KeypointDetectorNode()
