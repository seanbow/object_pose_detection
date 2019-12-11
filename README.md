# Object Pose Detection and Semantic Mapping
This package runs the full pipeline for object detection (based on `darknet_ros`), keypoint localization and semantic mapping.

## Packages
Except for [gtsam](gtsam), included here as separate dependencies, we include the following packages:
1. The [object_keypoint_detector](object_keypoint_detector) package is used to detect keypoints on provided images.
1. The [semslam](semslam/semslam) package is responsible for handling the semantic mapping process.
1. The [bag_extractor](bag_extractor) package is used to save data to timestamped images and `.npz` files that contain groundtruth poses, in order to add additional data to the training process.
1. The [object_pose_interface_msgs](object_pose_interface_msgs) and [semslam_msgs](semslam/semslam_msgs) packages contain the necessary ROS interface messages.
1. (Although not used, we also include the legacy [py_faster_rcnn_ros](py_faster_rcnn_ros) package for object detection and the [feature_tracker](semslam/feature_tracker) and [svo_msgs](semslam/svo_msgs) used for geometric feature tracking in the semantic SLAM pipeline.)

## Dependencies
1. The [semslam](semslam) package needs `libpng++-dev`.

## Usage
Depending on whether you want to use YOLO with Darknet, YOLO with PyTorch or PyFaster, modify [this](object_keypoint_detector/launch/object_pose_detection_yolo_darknet.launch), [this](object_keypoint_detector/launch/object_pose_detection_yolo_pytorch.launch) or [this](object_keypoint_detector/launch/object_pose_detection_pyfaster.launch) launch file, and:
1. Specify the path to your `num_keypoints_file` in the launch file.
1. Specify the path to your model in the launch file (parameter `model_path`).
1. Define your model type as `StackedHourglass` or `CPN50` in the launch file (parameter `model_type`).
1. Copy the files for the classes you used in your `num_keypoints_file` (and in your keypoint detection model) from the `objects_all` [directory](semslam/semslam/models/objects_all) to the `objects` [directory](semslam/semslam/models/objects).
1. Then, modify and run your [launch](semslam/semslam/launch) file from the [semslam](semslam/semslam) package.