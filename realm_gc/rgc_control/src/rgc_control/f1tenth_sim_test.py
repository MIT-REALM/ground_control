#!/usr/bin/env python3
"""Define class for robot control """
import os

import numpy as np
import rospy
from f1tenth_msgs.msg import F1TenthDriveStamped
from rgc_state_estimators.msg import F1TenthState

# from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy
from rgc_control.policies.common import F1TenthAction
from rgc_control.robot_control import RobotControl


class F1TenthControl(RobotControl):
    def __init__(self):
        super(F1TenthControl, self).__init__()

        # Publish cmd:[steering angle,acceleration] for state estimation
        self.control_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_0",
            # "/drive_1",
            F1TenthDriveStamped,
            queue_size=1,
        )
        self.desired_speed = 0.0
        self.control = F1TenthAction(0.0, 0.0)

        # Subscribe to state estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", f"{rospy.get_name()}/estimate"
        )
        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, F1TenthState, self.state_estimate_callback
        )

        rospy.sleep(2.0)  # additional waiting for state to converge
        rospy.loginfo("State estimate has converged :)")

    def state_estimate_callback(self, msg):
        self.state = msg

    def reset_control(self, msg=None):
        """Reset the control to stop the experiment and publish the command."""
        self.control = F1TenthAction(0.0, 0.0)
        msg = F1TenthDriveStamped()
        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration
        self.control_pub.publish(msg)
        self.desired_speed = 0.0

    def update(self):
        """
        Update and publish the control.
        This function implements and calls the control prediction and update steps.
        """
        msg = F1TenthDriveStamped()

        msg.drive.mode = 0
        msg.drive.speed = 1.0
        msg.drive.acceleration = 1.0
        msg.drive.steering_angle = (np.random.rand()-1/2) * 2/9 * np.pi
    
        self.control_pub.publish(msg)


if __name__ == "__main__":
    try:
        control_node = F1TenthControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
