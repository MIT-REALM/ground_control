#include "infer_v0.h"

#include <f1tenth_msgs/F1TenthDriveStamped.h>
#include <fmt/format.h>

#include <sophus/se2.hpp>

namespace rl_runtime {

namespace fs = std::filesystem;

InferV0::InferV0(ros::NodeHandle nh, const ros::NodeHandle& private_nh) {
    std::string trt_model_path;
    if (!private_nh.getParam("trt_model_path", trt_model_path)) {
        throw std::runtime_error("Parameter `trt_model_path` not found!");
    }

    if (!private_nh.getParam("servo_pos_offset", servo_pos_offset_)) {
        throw std::runtime_error("Parameter `servo_pos_offset` not found!");
    }
    if (!private_nh.getParam("servo_pos_gain", servo_pos_gain_)) {
        throw std::runtime_error("Parameter `servo_pos_gain` not found!");
    }
    if (!private_nh.getParam("v_coeff", v_coeff_)) {
        throw std::runtime_error("Parameter `v_coeff` not found!");
    }
    if (!private_nh.getParam("dt_min", dt_min_)) {
        throw std::runtime_error("Parameter `dt_min` not found!");
    }
    if (!private_nh.getParam("obs_lidar_theta_i", obs_lidar_theta_i_)) {
        throw std::runtime_error("Parameter `obs_lidar_theta_i` not found!");
    }
    if (!private_nh.getParam("obs_lidar_theta_f", obs_lidar_theta_f_)) {
        throw std::runtime_error("Parameter `obs_lidar_theta_f` not found!");
    }
    if (!private_nh.getParam("obs_n_lidar", obs_n_lidar_)) {
        throw std::runtime_error("Parameter `obs_n_lidar` not found!");
    }
    if (!private_nh.getParam("max_hit_dist", max_hit_dist_)) {
        throw std::runtime_error("Parameter `max_hit_dist` not found!");
    }

    trt_model_ = std::make_unique<TRTModel>(trt_model_path);

    cmd_pub_ = nh.advertise<f1tenth_msgs::F1TenthDriveStamped>("cmd", 1);
    lidar_pub_ = nh.advertise<sensor_msgs::LaserScan>("obs_lidar", 2);

    // Setup the scan debug message.
    scan_dbg_.angle_min = static_cast<float>(obs_lidar_theta_i_);
    scan_dbg_.angle_max = static_cast<float>(obs_lidar_theta_f_);
    scan_dbg_.angle_increment =
        static_cast<float>((obs_lidar_theta_f_ - obs_lidar_theta_i_) / (obs_n_lidar_ - 1));
    scan_dbg_.range_min = 0.0;
    scan_dbg_.range_max = max_hit_dist_;
    scan_dbg_.ranges = std::vector<float>(obs_n_lidar_);

    const auto nodelay = ros::TransportHints{}.tcpNoDelay(true);
    odom_sub_ = nh.subscribe("odom", 1, &InferV0::OdomCb, this, nodelay);
    scan_sub_ = nh.subscribe("scan", 1, &InferV0::ScanCb, this, nodelay);
    nav_goal_sub_ = nh.subscribe("nav_goal", 1, &InferV0::NavGoalCb, this, nodelay);

    last_run_stamp_ = ros::Time::now();
}

void InferV0::ScanCb(const sensor_msgs::LaserScanConstPtr& msg) {
    scan_ = msg;
    TryRunAndPublish();
}

void InferV0::OdomCb(const nav_msgs::OdometryConstPtr& msg) {
    // Maybe interface with the rviz stuff for determining the angle and stuff?
    odom_ = msg;
    TryRunAndPublish();
}

void InferV0::NavGoalCb(const geometry_msgs::PoseStampedConstPtr& msg) {
    nav_goal_ = msg;
    ROS_INFO("Received goal!");
    TryRunAndPublish();
}

void InferV0::TryRunAndPublish() {
    // Only try if we have both a scan and odom messages.
    if (scan_ == nullptr || odom_ == nullptr || nav_goal_ == nullptr) {
        return;
    }

    // If it has been too soon since the last message, return.
    const auto now = ros::Time::now();
    const auto time_since_last_run = now - last_run_stamp_;
    if (time_since_last_run.toSec() < dt_min_) {
        return;
    }

    // Otherwise, run.
    RunAndPublish();
}

Sophus::SE2d PoseToSE2(const geometry_msgs::Pose& pose) {
    const double yaw = std::atan2(pose.orientation.z, pose.orientation.w);
    return Sophus::SE2d{yaw, Sophus::Vector2d{pose.position.x, pose.position.y}};
}

void InferV0::RunAndPublish() {
    // Format the data into the observation vector.
    //          0: Horizontal error
    //     1 -  2: sin(theta)
    //          3: is_dead = 0
    //     4 - 25: lidar, spaced from [-π/2, π/2]
    //         26: body frame forward velocity
    const auto stamp_start = ros::Time::now();
    last_run_stamp_ = stamp_start;

    // 1: Compute X_tgt_robot.
    const Sophus::SE2d X_W_tgt = PoseToSE2(nav_goal_->pose);
    const Sophus::SE2d X_W_lidar = PoseToSE2(odom_->pose.pose);
    const auto X_lidar_tgt = X_W_lidar.inverse() * X_W_tgt;
    const auto R_tgt_lidar = X_lidar_tgt.so2().inverse();

    const auto err_horizontal = static_cast<float>(X_lidar_tgt.translation().y());
    const auto theta = static_cast<float>(R_tgt_lidar.log());

    trt_model_->input()[0] = err_horizontal;
    trt_model_->input()[1] = std::sin(theta);
    trt_model_->input()[2] = std::cos(theta);
    trt_model_->input()[3] = -1.f;  // is dead = 0. [0, 1] -> [-1, 1].
    trt_model_->input()[4 + obs_n_lidar_] = static_cast<float>(odom_->twist.twist.linear.x);

    // 2: Lidar observations.
    const float exp_min_max_hit = std::exp(-max_hit_dist_);

    const auto angle_min = scan_->angle_min;
    const auto angle_incr = scan_->angle_increment;
    for (int ii = 0; ii < obs_n_lidar_; ++ii) {
        const double frac_des = static_cast<double>(ii) / (obs_n_lidar_ - 1);
        const auto theta_des =
            (1.0 - frac_des) * obs_lidar_theta_i_ + frac_des * obs_lidar_theta_f_;
        const auto scan_idx = static_cast<int>((theta_des - angle_min) / angle_incr);

        // Clip between 0 and the max_dist used in the code.
        const auto scan_dist = std::clamp(scan_->ranges[scan_idx], 0.f, max_hit_dist_);
        auto lidar_obs = (std::exp(-scan_dist) - exp_min_max_hit) / (1 - exp_min_max_hit);
        trt_model_->input()[4 + ii] = 2.f * lidar_obs - 1.f;

        scan_dbg_.ranges[ii] = scan_dist;
    }

    // Output control: [v_desired, κ_desired]
    const auto& output = trt_model_->run();
    const auto stamp_model_done = ros::Time::now();

    const auto v_des = output[0];
    const auto kappa_des = output[1];

    // Convert kappa_des to a servo position.
    const auto servo_pos = CurvatureToServoPos(kappa_des);

    f1tenth_msgs::F1TenthDriveStamped::Ptr stamped_msg =
        boost::make_shared<f1tenth_msgs::F1TenthDriveStamped>();
    stamped_msg->header.seq = cmd_seq_++;
    stamped_msg->header.stamp = stamp_model_done;
    stamped_msg->drive.steer_mode = f1tenth_msgs::F1TenthDrive::MODE_SERVO;
    stamped_msg->drive.servo_position = servo_pos;

    stamped_msg->drive.mode = f1tenth_msgs::F1TenthDrive::MODE_SPEED;
    stamped_msg->drive.speed = v_coeff_ * v_des;

    cmd_pub_.publish(stamped_msg);

    ROS_INFO_STREAM(
        fmt::format("e: {:.1f}, th: {:.1f}, kappa: {:.1f}", err_horizontal, theta, kappa_des));

    // TODO: Maybe publish some diagnostics about execution time or something?

    // Publish the used laser scan.
    scan_dbg_.header.frame_id = scan_->header.frame_id;
    scan_dbg_.header.stamp = stamp_start;
    lidar_pub_.publish(scan_dbg_);
}

float InferV0::CurvatureToServoPos(float kappa) const {
    return servo_pos_gain_ * kappa + servo_pos_offset_;
}

}  // namespace rl_runtime
