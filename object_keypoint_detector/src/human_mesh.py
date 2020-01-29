#!/usr/bin/env python
import rospy
import rospkg

import time
from math import floor, ceil, sqrt
import cv2
import numpy as np
from os.path import join
import torch
import torchfile
from torchvision.transforms import Normalize


import message_filters
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from object_pose_interface_msgs.msg import ira_dets
from object_pose_interface_msgs.msg import KeypointDetection3D
from object_pose_interface_msgs.msg import KeypointDetections3D
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from models import hmr, SMPL

CAMERA_FOCAL_LENGTH_LOCAL = 5000.
CAMERA_FOCAL_LENGTH_GLOBAL = 679.6778564453125

class KeypointDetectorNode(object):
    ''' This node uses a stacked hourglass network implemented in Torch and converted to PyTorch to
    find the locations of keypoints in an image.
    '''
    def __init__(self):
        rospy.init_node('human_mesh_node')
        self.keypoints_pub = rospy.Publisher('pose_estimator/keypoints', KeypointDetections3D, queue_size=5)
        
        self.mesh_debug = rospy.Publisher('pose_estimator/mesh_debug', Image, queue_size=1)

        self.img_keypoints = rospy.Publisher('pose_estimator/img_keypoints', Image, queue_size=1)

        self.mesh_rviz = rospy.Publisher('pose_estimator/meshes', MarkerArray, queue_size=1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo("Using %s", self.device)

        self.model_type = rospy.get_param('~model_type', 'HMR')
        self.img_size = rospy.get_param('~img_size', 224)
        self.yolo = rospy.get_param('/yolo')
        self.debug = rospy.get_param('~debug', False)
        
        if rospy.has_param('~image_topic'):
            image_topic_set = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
            self.image_topic = image_topic_set
        else:
            rospy.logerr("No image topic specified")
        
        self.width = 0
        self.height = 0

        rospack = rospkg.RosPack()
        model_base_path = join(rospack.get_path('object_keypoint_detector'), 'models')

        model_path = rospy.get_param('~model_path', join(model_base_path, "model_checkpoint.pt"))

        self.model = hmr(join(rospack.get_path('object_keypoint_detector'), 'src', 'models', 'smpl', 'smpl_mean_params.npz'))
        checkpoint = torch.load(model_path)
        self.model.cuda()
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.smpl = SMPL(model_base_path, batch_size=1, create_transl=False).cuda()
        self.model.eval()
        rospy.loginfo("Loaded model")

        self.bridge = CvBridge()
        
        if (self.yolo):
            detections_sub = message_filters.Subscriber('detected_objects', BoundingBoxes)
        else:
            detections_sub = message_filters.Subscriber('detected_objects', ira_dets)

        image_sub = message_filters.Subscriber(self.image_topic, Image)
        combined_sub = message_filters.TimeSynchronizer([detections_sub, image_sub], 100)
        combined_sub.registerCallback(self.mesh_reconstruct)
        rospy.loginfo("Spinning")
        rospy.spin()
    

    def vert_faces_to_triangle_list(self, verts, faces):
        triangle_pts = verts[faces.ravel(),:]
        triangle_list = []

        for pt in triangle_pts:
            triangle_list.append(Point(*pt))

        return triangle_list


    def mesh_reconstruct(self, detections_msg, image_msg):
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

        self.img = (image/255.0 - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        # self.img = image/255.0
        self.img_preallocated[self.width+self.height//2:self.width+self.height//2+self.height,self.height+self.width//2:self.height+self.width//2+self.width,:] = self.img
        self.img_published = image.astype(np.uint8)
        
        if (self.yolo):
            bounding_boxes_detected = detections_msg.bounding_boxes
            bounding_boxes_detected = [detection for detection in bounding_boxes_detected if detection.Class=='person']
        else:
            bounding_boxes_detected = detections_msg.dets
            bounding_boxes_detected = [detection for detection in bounding_boxes_detected if detection.obj_name=='person']

        if (self.yolo):
            if len(bounding_boxes_detected) == 0:
                keypoint_detections = KeypointDetections3D()
                keypoint_detections.header = detections_msg.header
                self.keypoints_pub.publish(keypoint_detections)
                
                image_msg = self.bridge.cv2_to_imgmsg(self.img_published, encoding="rgb8")
                self.mesh_debug.publish(self.bridge.cv2_to_imgmsg(np.zeros([self.img_size,self.img_size,3]).astype(np.uint8), encoding="rgb8"))
                self.img_keypoints.publish(image_msg)
                return
        else:
            if len(bounding_boxes_detected) == 0:
                keypoint_detections = KeypointDetections3D()
                keypoint_detections.header = detections_msg.header
                self.keypoints_pub.publish(keypoint_detections)
                
                image_msg = self.bridge.cv2_to_imgmsg(self.img_published, encoding="rgb8")
                self.mesh_debug.publish(self.bridge.cv2_to_imgmsg(np.zeros([self.img_size,self.img_size,3]).astype(np.uint8), encoding="rgb8"))
                self.img_keypoints.publish(image_msg)
                return

        patches, bounds = self.get_patches_and_bounds(bounding_boxes_detected, self.img_preallocated)

        pred_vertices, pred_joints, camera_translation, camera_translation_local = self.get_keypoints(patches, bounds)

        if (self.debug):
            from utils.renderer import Renderer
            self.renderer = Renderer()
            img_mesh_published = self.renderer.render(pred_vertices[0], self.smpl.faces, camera_t=camera_translation_local[0], img=np.transpose(patches[0], (1,2,0))*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406], use_bg=True, body_color='light_blue')
            self.mesh_debug.publish(self.bridge.cv2_to_imgmsg((255.0*img_mesh_published).astype(np.uint8), encoding="rgb8"))

        keypoint_detections = KeypointDetections3D()
        markers_msg = MarkerArray()
        keypoint_detections.header = detections_msg.header

        for i, detection in enumerate(bounding_boxes_detected):
            detection_msg = KeypointDetection3D()
            obj_name = (detection.Class) if (self.yolo) else (detection.obj_name)
            detection_msg.obj_name = obj_name
            predictions = pred_joints[i, :, :] + camera_translation_local[i]
            predictions_publish = pred_joints[i, :, :] + camera_translation[i]

            vertices_translated = pred_vertices[i, :, :] + camera_translation[i]
            marker_msg = Marker()
            marker_msg.header.frame_id = "zed_left_camera_optical_frame"
            marker_msg.type = Marker.TRIANGLE_LIST
            marker_msg.action = Marker.ADD
            marker_msg.ns = "human_pose_demo"
            marker_msg.id = i
            marker_msg.lifetime = rospy.Time(2.0)
            marker_msg.scale.x = 1
            marker_msg.scale.y = 1
            marker_msg.scale.z = 1
            marker_msg.pose.orientation.w = 1
            marker_msg.color.r = 0.65098039
            marker_msg.color.g = 0.74117647
            marker_msg.color.b = 0.85882353
            marker_msg.color.a = 1
            # marker_msg.points = self.vert_faces_to_triangle_list(vertices_translated, self.smpl.faces)
            # markers_msg.markers.append(marker_msg)
                        
            for joint_i in predictions:
                coords = CAMERA_FOCAL_LENGTH_LOCAL*joint_i[:2]/joint_i[2] + self.img_size/2.0
                img_coords = [0, 0]
                img_coords[0] = bounds[i][0] + int(1.0 * coords[0] / self.img_size * (bounds[i][1] - bounds[i][0]) + 0.5)
                img_coords[1] = bounds[i][2] + int(1.0 * coords[1] / self.img_size * (bounds[i][3] - bounds[i][2]) + 0.5)
                                    
                cv2.circle(self.img_published, (img_coords[0], img_coords[1]), 5, (255, 0, 0), thickness=-1)
            
            for joint_publish_i in predictions_publish:
                detection_msg.x.append(joint_publish_i[0])
                detection_msg.y.append(joint_publish_i[1])
                detection_msg.z.append(joint_publish_i[2])
                                                    
            keypoint_detections.detections.append(detection_msg)

        self.keypoints_pub.publish(keypoint_detections)
        self.mesh_rviz.publish(markers_msg)

        image_msg = self.bridge.cv2_to_imgmsg(self.img_published, encoding="rgb8")
        self.img_keypoints.publish(image_msg)

        rospy.loginfo("Found keypoints for %d humans in %f seconds", len(bounding_boxes_detected), time.clock() - before_time)

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

    def get_keypoints(self, patches, bounds):
        ''' Runs the images through the network
        '''
        patches_tensor = torch.from_numpy(patches)
        patches_tensor = patches_tensor.to(self.device, dtype=torch.float32)
        bounds_tensor = torch.from_numpy(np.array(bounds))
        bounds_tensor = bounds_tensor.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = self.model(patches_tensor)
            pred_betas[:] = 0.
            pred_betas[0] = 1.
            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
            pred_joints = pred_joints[:,:24]
            
            # Figure out translations
            # One local translation, for each bounding box
            camera_translation_local = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*CAMERA_FOCAL_LENGTH_LOCAL/(self.img_size * pred_camera[:,0] +1e-9)],dim=-1)
            # One global translation, moving the people to the global frame
            w = bounds_tensor[:,1] - bounds_tensor[:,0]
            h = bounds_tensor[:,3] - bounds_tensor[:,2]
            a = torch.max(w,h)
            depth = 2*CAMERA_FOCAL_LENGTH_GLOBAL/(a * pred_camera[:,0] +1e-9)
            center_x = (bounds_tensor[:,1] + bounds_tensor[:,0])/2
            center_y = (bounds_tensor[:,3] + bounds_tensor[:,2])/2
            camera_translation = torch.stack([camera_translation_local[:,0]+(depth/CAMERA_FOCAL_LENGTH_GLOBAL)*(center_x-self.width/2), camera_translation_local[:,1]+(depth/CAMERA_FOCAL_LENGTH_GLOBAL)*(center_y-self.height/2), depth],dim=-1)
        # The network returns a list of the outputs of all of the hourglasses
        # in the stack.  We want the output of the final hourglass
        return pred_vertices.cpu().numpy(), pred_joints.cpu().numpy(), camera_translation.cpu().numpy(), camera_translation_local.cpu().numpy()

if __name__ == '__main__':
    KeypointDetectorNode()
