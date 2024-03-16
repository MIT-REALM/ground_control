#!/usr/bin/env python3
"""Define a state estimator for the turtlebot using a Moving Average Filter."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from transforms3d.euler import quat2euler

from rgc_state_estimators.msg import TurtlebotState
from rgc_state_estimators.state_estimator import StateEstimator


class TurtlebotMAFStateEstimator(StateEstimator):
    """
    State estimator for the Turtlebot using a Moving Average Filter.
    """

    def __init__(self):
        super(TurtlebotMAFStateEstimator, self).__init__()

        # Fetch additional parameters
        self.decay_rate = rospy.get_param("~decay_rate", 0.75)
        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_turtle0/realm_turtle0"
        )

        # Set a timer to poll the parameters
        self.poll_timer = rospy.Timer(
            rospy.Duration(1.0), self.poll_parameters_callback
        )

        # Initialize the MAF variables
        self.state = np.zeros((3, 1))  # [x, y, theta]

        # Set up subscribers
        self.last_position_msg = None
        self.position_sub = rospy.Subscriber(
            self.position_topic, TransformStamped, self.position_callback
        )

        # Publisher for PoseStamped messages
        self.estimate_pub = rospy.Publisher(
            f"{rospy.get_name()}/estimate", TurtlebotState, queue_size=10
        )

    def poll_parameters_callback(self, _):
        """Poll the parameters from the parameter server."""
        new_decay_rate = rospy.get_param("~decay_rate", 0.75)

        if new_decay_rate != self.decay_rate:
            self.decay_rate = new_decay_rate
            rospy.loginfo(f"Updated decay rate to {self.decay_rate}")

    def reset_state(self, msg=None):
        """Reset the state of the MAF."""
        self.state = np.zeros((3, 1))  # Reset state to zeros

    def position_callback(self, msg):
        """
        Update the state based on new position measurements.
        Placeholder function - implement measurement update logic here.
        """
        self.last_position_msg = msg

    def update(self):
        """
        Update the filter state and publish the new state estimate.
        This function should implement or call the MAF prediction and update steps.
        """
        if self.last_position_msg is not None:
            # Extract position measurements from the message
            x = self.last_position_msg.transform.translation.x
            y = self.last_position_msg.transform.translation.y

            # Convert quaternion to yaw angle
            (qx, qy, qz, qw) = (
                self.last_position_msg.transform.rotation.x,
                self.last_position_msg.transform.rotation.y,
                self.last_position_msg.transform.rotation.z,
                self.last_position_msg.transform.rotation.w,
            )
            _, _, theta = quat2euler([qw, qx, qy, qz])

            measured_state = np.array([x, y, theta]).reshape(-1, 1)

            # Update the filter state
            self.state = (
                self.decay_rate * self.state + (1 - self.decay_rate) * measured_state
            )

            self.last_position_msg = None

        # Publish the new state estimate
        msg = TurtlebotState()
        msg.x = self.state[0, 0]
        msg.y = self.state[1, 0]
        msg.theta = self.state[2, 0]
        self.estimate_pub.publish(msg)


if __name__ == "__main__":
    try:
        maf_node = TurtlebotMAFStateEstimator()
        maf_node.run()
    except rospy.ROSInterruptException:
        pass
