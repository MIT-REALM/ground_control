#include "pose2odom.h"

#include <nav_msgs/Odometry.h>

#include <algorithm>
#include <sophus/se2.hpp>

namespace pose2odom {
Pose2Odom::Pose2Odom(ros::NodeHandle nh, const ros::NodeHandle& private_nh) {
    if (!private_nh.getParam("vel_est_dt", vel_est_dt_)) {
        throw std::runtime_error("Parameter `vel_est_dt` not provided!");
    }

    int buf_capacity;
    if (!private_nh.getParam("buf_capacity", buf_capacity)) {
        throw std::runtime_error("Parameter `buf_capacity` not provided!");
    }

    pose_buf_ = boost::circular_buffer<geometry_msgs::PoseStampedConstPtr>(buf_capacity);

    odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 1);
    const auto nodelay = ros::TransportHints{}.tcpNoDelay(true);
    pose_sub_ = nh.subscribe("pose", 1, &Pose2Odom::poseCb, this, nodelay);
}

Sophus::SE2d PoseToSE2(geometry_msgs::Pose pose) {
    const double yaw = std::atan2(pose.orientation.z, pose.orientation.w);
    return Sophus::SE2d{yaw, Sophus::Vector2d{pose.position.x, pose.position.y}};
}

Sophus::Vector3d Pose2Odom::EstimateTwist(const geometry_msgs::PoseStamped& msg) {
    // If this is the first message, then zero twist.
    if (pose_buf_.empty()) {
        return {};
    }

    // Otherwise, find the message with the closest timestamp to the desired dt.
    const auto target_stamp = msg.header.stamp - ros::Duration(vel_est_dt_);
    const auto compare = [target_stamp](const geometry_msgs::PoseStampedConstPtr& a,
                                        const geometry_msgs::PoseStampedConstPtr& b) {
        return (a->header.stamp - target_stamp).toSec() < (b->header.stamp - target_stamp).toSec();
    };
    const auto& prev_pose_it =
        std::min_element(std::begin(pose_buf_), std::end(pose_buf_), compare);
    const auto& prev_pose = *prev_pose_it;

    const auto dt = (msg.header.stamp - prev_pose->header.stamp).toSec();
    const Sophus::SE2d pose_curr = PoseToSE2(msg.pose);
    const Sophus::SE2d pose_prev = PoseToSE2(prev_pose->pose);
    // [vx vy w]
    Sophus::Vector3d twist_est = (pose_curr * pose_prev.inverse()).log() / dt;
    return twist_est;
}

void Pose2Odom::poseCb(const geometry_msgs::PoseStampedConstPtr& msg) {
    // Estimate twist.
    const auto twist_est = EstimateTwist(*msg);

    // Fill odom message.
    auto odom = boost::make_shared<nav_msgs::Odometry>();
    odom->header = msg->header;
    odom->pose.pose = msg->pose;
    odom->twist.twist.linear.x = twist_est[0];
    odom->twist.twist.linear.y = twist_est[1];
    odom->twist.twist.angular.z = twist_est[2];

    // Store in buffer for use in computing velocity estimates.
    pose_buf_.push_back(msg);

    // Publish odom.
    odom_pub_.publish(odom);
}

}  // namespace pose2odom