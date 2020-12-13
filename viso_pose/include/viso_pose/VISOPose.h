#include <deque>

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <png++/png.hpp>
#include <viso_stereo.h>

#include <fstream>

class VISOPose
{
  public:
    VISOPose(double px_sigma);

    void setExtrinsics(const std::vector<double>& I_q_C,
                       const std::vector<double>& I_p_C);

    void img0_callback(const sensor_msgs::ImageConstPtr& msg);
    void img1_callback(const sensor_msgs::ImageConstPtr& msg);

    void caminfo_callback(const sensor_msgs::CameraInfo::ConstPtr& msg0,
                          const sensor_msgs::CameraInfo::ConstPtr& msg1);

    void tryProcessNextImages();

    void publishTransform(const Eigen::Matrix4d& pose);

  private:
    std::deque<sensor_msgs::ImageConstPtr> img0_queue_;
    std::deque<sensor_msgs::ImageConstPtr> img1_queue_;

    double px_sigma_;

    int64_t last_img0_seq_, last_img1_seq_;

    VisualOdometryStereo::parameters viso_params_;
    bool got_calibration_;

    boost::shared_ptr<VisualOdometryStereo> viso_;

    Matrix pose_;
    Matrix full_pose_; // for debugging & publishing trajectory

    Eigen::Matrix4d I_T_C_;
    Eigen::Matrix4d C_T_I_;

    ros::NodeHandle nh_, pnh_;
    ros::Publisher marker_pub_;
    ros::Publisher relp_pub_;
    ros::Publisher clock_pub_;

    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>
      caminfo0_sub_;
    boost::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>
      caminfo1_sub_;
    boost::shared_ptr<
      message_filters::TimeSynchronizer<sensor_msgs::CameraInfo,
                                        sensor_msgs::CameraInfo>>
      caminfo_sync_;

    std::vector<geometry_msgs::Point> trajectory_;

    ros::Time last_img_time_;
    Eigen::Matrix4d last_img_pose_;

    bool initialized_;

    std::ofstream out_file_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

void
visoMatrixToEigen(const Matrix& viso, Eigen::Matrix4d& eigen)
{
    for (size_t i = 0; i < viso.m; ++i) {
        for (size_t j = 0; j < viso.n; ++j) {
            eigen(i, j) = viso.val[i][j];
        }
    }
}