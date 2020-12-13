#include <ros/ros.h>

#include "viso_pose/VISOPose.h"

int
main(int argc, char* argv[])
{
    ros::init(argc, argv, "viso_node");

    ros::NodeHandle nh("~");

    double px_sigma;

    if (!nh.getParam("viso_px_sigma", px_sigma)) {
        ROS_ERROR_STREAM("Unable to read parameter viso_px_sigma");
        return 1;
    }

    VISOPose viso(px_sigma);

    ros::Subscriber cam0_sub =
      nh.subscribe("/cam0/image_raw", 1000, &VISOPose::img0_callback, &viso);

    ros::Subscriber cam1_sub =
      nh.subscribe("/cam1/image_raw", 1000, &VISOPose::img1_callback, &viso);

    // Check for any given extrinsic calibration information
    std::vector<double> I_p_C, I_q_C;

    if (nh.getParam("I_p_C", I_p_C) && nh.getParam("I_q_C", I_q_C)) {
        viso.setExtrinsics(I_q_C, I_p_C);
    } else {
        ROS_WARN_STREAM(
          "Running VISO Pose without extrinsic calibration data.");
    }

    // ros::Subscriber key_sub = nh.subscribe("/klt/keyframe_features",
    // 									   1000,
    // 									   &VISOPose::keyframeCallback,
    // 									   &viso);

    ros::spin();
}