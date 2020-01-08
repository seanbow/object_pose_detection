# Object Pose Detection and Semantic Mapping
This package runs the full pipeline for object detection (based on `darknet_ros`), keypoint localization and semantic mapping.

## Packages
1. The [object_keypoint_detector](object_keypoint_detector) package is used to detect keypoints on provided images.
1. The [semantic_slam](https://github.com/seanbow/semantic_slam) package is responsible for handling the semantic mapping process.
1. The [bag_extractor](bag_extractor) package is used to save data to timestamped images and `.npz` files that contain groundtruth poses, in order to add additional data to the training process.
1. The [object_pose_interface_msgs](object_pose_interface_msgs) package contains the necessary ROS interface messages.

## Installing dependencies
Scripts to install all required dependencies are available at https://github.com/seanbow/xavier-setup-scripts (contrary to the name, these scripts function correctly both on desktop- and nVidia xavier- based platforms).

Alternatively,
1. Install [ROS](https://www.ros.org/) -- only tested thus far on ROS Melodic
1. `sudo apt install libgoogle-glog-dev libpng++-dev ros-melodic-rosfmt`
1. Build and install Google's `ceres-solver` from source: https://github.com/ceres-solver/ceres-solver
    - Be sure to set `-DCMAKE_C_FLAGS="-march=native" -DCMAKE_CXX_FLAGS="-march=native"` when calling CMake or else you may run into memory alignment related issues and crashes
1. Build and install GTSAM: https://github.com/borglab/gtsam. Make sure GTSAM_USE_SYSTEM_EIGEN and GTSAM_TYPEDEF_POINTS_TO_VECTORS are set to true.

## Usage
### Semantic SLAM
1. Modify [this](https://github.com/seanbow/semantic_slam/blob/master/launch/semantic_slam.launch) launch file to:
    1. specify the path to your `num_keypoints_file` in the launch file.
    1. specify the path to your model in the launch file (parameter `model_path`).
    1. define your model type as `StackedHourglass` or `CPN50` in the launch file (parameter `model_type`).
1. Copy the files for the classes you used in your `num_keypoints_file` (and in your keypoint detection model) from the `objects_all` [directory](https://github.com/seanbow/semantic_slam/tree/master/models/objects_all) to the `objects` [directory](https://github.com/seanbow/semantic_slam/tree/master/models/objects).
1. Run the launch file.

### Human mesh estimation
1. Copy your models in the `models` [directory](object_keypoint_detector/models). You will need a `.pt` and a `.pkl` model.
1. Modify [this](object_keypoint_detector/launch/human_mesh_darknet.launch) launch file to point to the right model and camera topic.
1. Run the launch file.
