# Object Pose Detection and Semantic Mapping
This package runs the full pipeline for object detection (based on `darknet_ros`), keypoint localization and semantic mapping.

## Packages
1. The [object_keypoint_detector](object_keypoint_detector) package is used to detect keypoints on provided images.
1. The [semantic_slam](semantic_slam) package is responsible for handling the semantic mapping process.
1. The [bag_extractor](bag_extractor) package is used to save data to timestamped images and `.npz` files that contain groundtruth poses, in order to add additional data to the training process.
1. The [object_pose_interface_msgs](object_pose_interface_msgs) package contains the necessary ROS interface messages.

## Installing dependencies
1. Install [ROS](https://www.ros.org/) -- only tested thus far on ROS Melodic
1. `sudo apt install libgoogle-glog-dev libpng++-dev ros-melodic-rosfmt`
1. Build and install Google's `ceres-solver` from source: https://github.com/ceres-solver/ceres-solver
1. Build and install GTSAM: https://github.com/borglab/gtsam. Make sure GTSAM_USE_SYSTEM_EIGEN is set to true

## Usage
1. Modify [this](semantic_slam/launch/semantic_slam.launch) launch file to:
    1. specify the path to your `num_keypoints_file` in the launch file.
    1. specify the path to your model in the launch file (parameter `model_path`).
    1. define your model type as `StackedHourglass` or `CPN50` in the launch file (parameter `model_type`).
1. Copy the files for the classes you used in your `num_keypoints_file` (and in your keypoint detection model) from the `objects_all` [directory](semantic_slam/models/objects_all) to the `objects` [directory](semantic_slam/models/objects).
1. Run the launch file.
