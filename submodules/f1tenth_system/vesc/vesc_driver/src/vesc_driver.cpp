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

#include "vesc_driver/vesc_driver.h"

#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <vesc_msgs/VescImuStamped.h>
#include <vesc_msgs/VescStateStamped.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace vesc_driver {

using std::placeholders::_1;

VescDriver::VescDriver(ros::NodeHandle nh, ros::NodeHandle private_nh)
    : vesc_(std::string(), std::bind(&VescDriver::vescPacketCallback, this, _1),
            std::bind(&VescDriver::vescErrorCallback, this, _1)),
      duty_cycle_limit_(private_nh, "duty_cycle", -1.0, 1.0),
      current_limit_(private_nh, "current"),
      brake_limit_(private_nh, "brake"),
      speed_limit_(private_nh, "speed"),
      position_limit_(private_nh, "position"),
      servo_limit_(private_nh, "servo", 0.0, 1.0),
      driver_mode_(MODE_INITIALIZING),
      fw_version_major_(-1),
      fw_version_minor_(-1) {
    // get vesc serial port address
    std::string port;
    if (!private_nh.getParam("port", port)) {
        ROS_FATAL("VESC communication port parameter required.");
        ros::shutdown();
        return;
    }

    // attempt to connect to the serial port
    try {
        vesc_.connect(port);
    } catch (SerialException e) {
        ROS_FATAL("Failed to connect to the VESC, %s.", e.what());
        ros::shutdown();
        return;
    }

    // create vesc state (telemetry) publisher
    state_pub_ = nh.advertise<vesc_msgs::VescStateStamped>("sensors/core", 10);
    imu_pub_ = nh.advertise<vesc_msgs::VescImuStamped>("sensors/imu", 10);
    imu_std_pub_ = nh.advertise<sensor_msgs::Imu>("sensors/imu_std", 10);

    // since vesc state does not include the servo position, publish the commanded
    // servo position as a "sensor"
    servo_sensor_pub_ = nh.advertise<std_msgs::Float64>("sensors/servo_position_command", 10);

    // subscribe to motor and servo command topics
    const auto nodelay = ros::TransportHints{}.tcpNoDelay(true);
    vesc_input_sub_ = nh.subscribe("commands", 10, &VescDriver::vescInputCallback, this, nodelay);

    // create a 50Hz timer, used for state machine & polling VESC telemetry
    timer_ = nh.createTimer(ros::Duration(1.0 / 50.0), &VescDriver::timerCallback, this);
}

/* TODO or TO-THINKABOUT LIST
  - what should we do on startup? send brake or zero command?
  - what to do if the vesc interface gives an error?
  - check version number against know compatable?
  - should we wait until we receive telemetry before sending commands?
  - should we track the last motor command
  - what to do if no motor command received recently?
  - what to do if no servo command received recently?
  - what is the motor safe off state (0 current?)
  - what to do if a command parameter is out of range, ignore?
  - try to predict vesc bounds (from vesc config) and command detect bounds errors
*/

void VescDriver::timerCallback(const ros::TimerEvent& event) {
    // VESC interface should not unexpectedly disconnect, but test for it anyway
    if (!vesc_.isConnected()) {
        ROS_FATAL("Unexpectedly disconnected from serial port.");
        timer_.stop();
        ros::shutdown();
        return;
    }

    /*
     * Driver state machine, modes:
     *  INITIALIZING - request and wait for vesc version
     *  OPERATING - receiving commands from subscriber topics
     */
    if (driver_mode_ == MODE_INITIALIZING) {
        // request version number, return packet will update the internal version numbers
        vesc_.requestFWVersion();
        if (fw_version_major_ >= 0 && fw_version_minor_ >= 0) {
            ROS_INFO("Connected to VESC with firmware version %d.%d", fw_version_major_,
                     fw_version_minor_);
            driver_mode_ = MODE_OPERATING;
        }
    } else if (driver_mode_ == MODE_OPERATING) {
        // poll for vesc state (telemetry)
        vesc_.requestState();
        // poll for vesc imu
        vesc_.requestImuData();
    } else {
        // unknown mode, how did that happen?
        assert(false && "unknown driver mode");
    }
}

void VescDriver::vescPacketCallback(const std::shared_ptr<VescPacket const>& packet) {
    if (packet->name() == "Values") {
        std::shared_ptr<VescPacketValues const> values =
            std::dynamic_pointer_cast<VescPacketValues const>(packet);

        vesc_msgs::VescStateStamped::Ptr state_msg(new vesc_msgs::VescStateStamped);
        state_msg->header.stamp = ros::Time::now();
        state_msg->state.temp_fet = values->temp_fet();
        state_msg->state.temp_motor = values->temp_motor();
        state_msg->state.current_motor = values->avg_motor_current();
        state_msg->state.current_input = values->avg_input_current();
        state_msg->state.avg_id = values->avg_id();
        state_msg->state.avg_iq = values->avg_iq();
        state_msg->state.duty_cycle = values->duty_cycle_now();
        state_msg->state.speed = values->rpm();
        state_msg->state.voltage_input = values->v_in();

        state_msg->state.charge_drawn = values->amp_hours();
        state_msg->state.charge_regen = values->amp_hours_charged();
        state_msg->state.energy_drawn = values->watt_hours();
        state_msg->state.energy_regen = values->watt_hours_charged();
        state_msg->state.displacement = values->tachometer();
        state_msg->state.distance_traveled = values->tachometer_abs();
        state_msg->state.fault_code = values->fault_code();

        state_msg->state.pid_pos_now = values->pid_pos_now();
        state_msg->state.controller_id = values->controller_id();

        state_msg->state.ntc_temp_mos1 = values->temp_mos1();
        state_msg->state.ntc_temp_mos2 = values->temp_mos2();
        state_msg->state.ntc_temp_mos3 = values->temp_mos3();
        state_msg->state.avg_vd = values->avg_vd();
        state_msg->state.avg_vq = values->avg_vq();

        state_pub_.publish(state_msg);
    } else if (packet->name() == "FWVersion") {
        std::shared_ptr<VescPacketFWVersion const> fw_version =
            std::dynamic_pointer_cast<VescPacketFWVersion const>(packet);
        // todo: might need lock here
        fw_version_major_ = fw_version->fwMajor();
        fw_version_minor_ = fw_version->fwMinor();
    } else if (packet->name() == "ImuData") {
        std::shared_ptr<VescPacketImu const> imuData =
            std::dynamic_pointer_cast<VescPacketImu const>(packet);

        auto imu_msg = boost::make_shared<vesc_msgs::VescImuStamped>();
        auto std_imu_msg = boost::make_shared<sensor_msgs::Imu>();
        imu_msg->header.stamp = ros::Time::now();
        std_imu_msg->header.stamp = ros::Time::now();

        imu_msg->imu.ypr.x = imuData->roll();
        imu_msg->imu.ypr.y = imuData->pitch();
        imu_msg->imu.ypr.z = imuData->yaw();

        imu_msg->imu.linear_acceleration.x = imuData->acc_x();
        imu_msg->imu.linear_acceleration.y = imuData->acc_y();
        imu_msg->imu.linear_acceleration.z = imuData->acc_z();

        imu_msg->imu.angular_velocity.x = imuData->gyr_x();
        imu_msg->imu.angular_velocity.y = imuData->gyr_y();
        imu_msg->imu.angular_velocity.z = imuData->gyr_z();

        imu_msg->imu.compass.x = imuData->mag_x();
        imu_msg->imu.compass.y = imuData->mag_y();
        imu_msg->imu.compass.z = imuData->mag_z();

        imu_msg->imu.orientation.w = imuData->q_w();
        imu_msg->imu.orientation.x = imuData->q_x();
        imu_msg->imu.orientation.y = imuData->q_y();
        imu_msg->imu.orientation.z = imuData->q_z();

        // Convert from g's -> m/s^2.
        constexpr auto g = 9.80665;
        std_imu_msg->linear_acceleration.x = g * imuData->acc_x();
        std_imu_msg->linear_acceleration.y = g * imuData->acc_y();
        std_imu_msg->linear_acceleration.z = g * imuData->acc_z();

        std_imu_msg->angular_velocity.x = imuData->gyr_x();
        std_imu_msg->angular_velocity.y = imuData->gyr_y();
        std_imu_msg->angular_velocity.z = imuData->gyr_z();

        std_imu_msg->orientation.w = imuData->q_w();
        std_imu_msg->orientation.x = imuData->q_x();
        std_imu_msg->orientation.y = imuData->q_y();
        std_imu_msg->orientation.z = imuData->q_z();

        imu_pub_.publish(imu_msg);
        imu_std_pub_.publish(std_imu_msg);
    }
}

void VescDriver::vescErrorCallback(const std::string& error) { ROS_ERROR("%s", error.c_str()); }

void VescDriver::vescInputCallback(const vesc_msgs::VescInputStamped::ConstPtr& msg) {
    if (driver_mode_ != MODE_OPERATING) {
        return;
    }

    // Motor command.
    const auto& cmd = msg->input;
    switch (cmd.type) {
        default:
        case vesc_msgs::VescInput::TYPE_RPM:
            vesc_.setSpeed(speed_limit_.clip(cmd.rpm));
            break;
        case vesc_msgs::VescInput::TYPE_DUTY:
            vesc_.setDutyCycle(duty_cycle_limit_.clip(cmd.duty));
            break;
        case vesc_msgs::VescInput::TYPE_CURRENT:
            vesc_.setCurrent(current_limit_.clip(cmd.current));
            break;
        case vesc_msgs::VescInput::TYPE_CURRENT_BRAKE:
            vesc_.setBrake(brake_limit_.clip(cmd.current_brake));
            break;
    }

    // Always send the servo command.
    const double servo_clipped = servo_limit_.clip(cmd.servo_position);
    vesc_.setServo(servo_clipped);

    // publish clipped servo value as a "sensor"
    auto servo_sensor_msg = boost::make_shared<std_msgs::Float64>();
    servo_sensor_msg->data = servo_clipped;
    servo_sensor_pub_.publish(servo_sensor_msg);
}

VescDriver::CommandLimit::CommandLimit(const ros::NodeHandle& nh, std::string str,
                                       const boost::optional<double>& min_lower,
                                       const boost::optional<double>& max_upper)
    : name(std::move(str)) {
    // check if user's minimum value is outside of the range min_lower to max_upper
    double param_min;
    if (nh.getParam(name + "_min", param_min)) {
        if (min_lower && param_min < *min_lower) {
            lower = *min_lower;
            ROS_WARN_STREAM("Parameter " << name << "_min (" << param_min
                                         << ") is less than the feasible minimum (" << *min_lower
                                         << ").");
        } else if (max_upper && param_min > *max_upper) {
            lower = *max_upper;
            ROS_WARN_STREAM("Parameter " << name << "_min (" << param_min
                                         << ") is greater than the feasible maximum (" << *max_upper
                                         << ").");
        } else {
            lower = param_min;
        }
    } else if (min_lower) {
        lower = *min_lower;
    }

    // check if the uers' maximum value is outside of the range min_lower to max_upper
    double param_max;
    if (nh.getParam(name + "_max", param_max)) {
        if (min_lower && param_max < *min_lower) {
            upper = *min_lower;
            ROS_WARN_STREAM("Parameter " << name << "_max (" << param_max
                                         << ") is less than the feasible minimum (" << *min_lower
                                         << ").");
        } else if (max_upper && param_max > *max_upper) {
            upper = *max_upper;
            ROS_WARN_STREAM("Parameter " << name << "_max (" << param_max
                                         << ") is greater than the feasible maximum (" << *max_upper
                                         << ").");
        } else {
            upper = param_max;
        }
    } else if (max_upper) {
        upper = *max_upper;
    }

    // check for min > max
    if (upper && lower && *lower > *upper) {
        ROS_WARN_STREAM("Parameter " << name << "_max (" << *upper << ") is less than parameter "
                                     << name << "_min (" << *lower << ").");
        double temp(*lower);
        lower = *upper;
        upper = temp;
    }

    std::ostringstream oss;
    oss << "  " << name << " limit: ";
    if (lower)
        oss << *lower << " ";
    else
        oss << "(none) ";
    if (upper)
        oss << *upper;
    else
        oss << "(none)";
    ROS_DEBUG_STREAM(oss.str());
}

double VescDriver::CommandLimit::clip(double value) {
    if (lower && value < lower) {
        ROS_INFO_THROTTLE(10, "%s command value (%f) below minimum limit (%f), clipping.",
                          name.c_str(), value, *lower);
        return *lower;
    }
    if (upper && value > upper) {
        ROS_INFO_THROTTLE(10, "%s command value (%f) above maximum limit (%f), clipping.",
                          name.c_str(), value, *upper);
        return *upper;
    }
    return value;
}

}  // namespace vesc_driver
