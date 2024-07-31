#!/usr/bin/env python3
"""Define control for turtlebot using LQR """
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from rgc_state_estimators.msg import TurtlebotState

from rgc_control.policies.common import TurtlebotAction

from rgc_control.robot_control import RobotControl


class TurtlebotControl(RobotControl):
    def __init__(self):
        super(TurtlebotControl, self).__init__()

        # Publish cmd velocity for state estimation
        self.control_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.control = TurtlebotAction(0.0, 0.0)

        # Subscribe to State estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", "turtlebot_state_estimator/estimate"
        )
        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, TurtlebotState, self.state_estimate_callback
        )

    def state_estimate_callback(self, msg):
        self.state = msg

    def stop_control_callback(self, msg):
        super().stop_control_callback(msg)

    def reset_control(self, msg=None):
        """Reset the turtlebot to its start position."""
        # Stop if no state information
        self.control = TurtlebotAction(0.0, 0.0)

        msg = Twist()
        msg.linear.x = self.control.linear_velocity
        msg.angular.z = self.control.angular_velocity
        self.control_pub.publish(msg)

    def update(self):
        """
        Update and publish the control.
        This function implements and calls the control prediction and update steps.
        """
        msg = Twist()
        msg.linear.x = 1.0
        msg.angular.z = np.random.rand()-1/2
        self.control_pub.publish(msg)


if __name__ == "__main__":
    try:
        control_node = TurtlebotControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
