// Copyright 2020 F1TENTH Foundation
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
//    and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used
//    to endorse or promote products derived from this software without specific prior
//    written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// -*- mode:c++; fill-column: 100; -*-

#include "vesc_ackermann/ackermann_to_vesc.h"

#include <std_msgs/Float64.h>
#include <vesc_msgs/VescInputStamped.h>

namespace vesc_ackermann {

template <typename T>
inline bool getRequiredParam(const ros::NodeHandle& nh, const std::string& name, T* value);

AckermannToVesc::AckermannToVesc(ros::NodeHandle nh, ros::NodeHandle private_nh) {
    // get conversion parameters
    if (!getRequiredParam(nh, "speed_to_erpm_gain", &speed_to_erpm_gain_)) return;
    if (!getRequiredParam(nh, "speed_to_erpm_offset", &speed_to_erpm_offset_)) return;
    if (!getRequiredParam(nh, "accel_to_current_gain", &accel_to_current_gain_)) return;
    if (!getRequiredParam(nh, "accel_to_brake_gain", &accel_to_brake_gain_)) return;
    if (!getRequiredParam(nh, "steering_angle_to_servo_gain", &steering_to_servo_gain_)) return;
    if (!getRequiredParam(nh, "steering_angle_to_servo_offset", &steering_to_servo_offset_)) return;

    // create publishers to vesc electric-RPM (speed) and servo commands
    vesc_pub_ = nh.advertise<vesc_msgs::VescInputStamped>("commands", 10);

    // subscribe to ackermann topic
    const auto nodelay = ros::TransportHints{}.tcpNoDelay(true);
    f1tenth_sub_ = nh.subscribe("f1tenth_cmd", 10, &AckermannToVesc::f1TenthCmdCallback, this, nodelay);
}

uint8_t to_vesc_type(const f1tenth_msgs::F1TenthDrive& msg) {
    if (msg.mode == f1tenth_msgs::F1TenthDrive::MODE_ACCEL) {
        if (msg.acceleration < 0) {
            return vesc_msgs::VescInput::TYPE_CURRENT_BRAKE;
        }
        return vesc_msgs::VescInput::TYPE_CURRENT;
    }

    // msg.mode == f1tenth_msgs::F1TenthDrive::MODE_SPEED
    return vesc_msgs::VescInput::TYPE_RPM;
}

double AckermannToVesc::GetServoPosition(const f1tenth_msgs::F1TenthDrive& drive) const {
    if (drive.steer_mode == f1tenth_msgs::F1TenthDrive::MODE_STEER_ANGLE) {
        return steering_to_servo_gain_ * drive.steering_angle + steering_to_servo_offset_;
    } else {
        return drive.servo_position;
    }
}

void AckermannToVesc::f1TenthCmdCallback(const f1tenth_msgs::F1TenthDriveStamped::ConstPtr& cmd) {
    // velocity -> ERPM.
    const double erpm = speed_to_erpm_gain_ * cmd->drive.speed + speed_to_erpm_offset_;

    // acceleration -> current.
    const double current = accel_to_current_gain_ * std::max(cmd->drive.acceleration, 0.f);
    const double current_brake = accel_to_brake_gain_ * std::min(cmd->drive.acceleration, 0.f);

    // steering angle -> servo angle.
    const double servo_position = GetServoPosition(cmd->drive);

    if (!ros::ok()) {
        return;
    }

    // Publish.
    vesc_msgs::VescInputStamped::Ptr vesc_msg = boost::make_shared<vesc_msgs::VescInputStamped>();
    vesc_msg->header = cmd->header;

    vesc_msg->input.type = to_vesc_type(cmd->drive);
    vesc_msg->input.rpm = erpm;
    vesc_msg->input.duty = 0.0;
    vesc_msg->input.current = current;
    vesc_msg->input.current_brake = current_brake;
    vesc_msg->input.servo_position = servo_position;

    vesc_pub_.publish(vesc_msg);
}

template <typename T>
inline bool getRequiredParam(const ros::NodeHandle& nh, const std::string& name, T* value) {
    if (nh.getParam(name, *value)) return true;

    ROS_FATAL("AckermannToVesc: Parameter %s is required.", name.c_str());
    return false;
}

}  // namespace vesc_ackermann
