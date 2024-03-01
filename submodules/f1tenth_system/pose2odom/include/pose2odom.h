#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <sophus/types.hpp>
#include <ros/ros.h>

#include <boost/circular_buffer.hpp>

namespace pose2odom {

class Pose2Odom {
   public:
    Pose2Odom(ros::NodeHandle nh, const ros::NodeHandle& private_nh);

   private:
    void poseCb(const geometry_msgs::PoseStampedConstPtr& msg);
    Sophus::Vector3d EstimateTwist(const geometry_msgs::PoseStamped& msg);

    boost::circular_buffer<geometry_msgs::PoseStampedConstPtr> pose_buf_;

    double vel_est_dt_;
    ros::Subscriber pose_sub_;
    ros::Publisher odom_pub_;
};
}  // namespace pose2odom
