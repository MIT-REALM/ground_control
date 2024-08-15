#!/usr/bin/env python3
"""Define a state estimator for the F1tenth using a Moving Average Filter."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from transforms3d.euler import quat2euler

from rgc_state_estimators.msg import F1TenthState
from rgc_state_estimators.state_estimator import StateEstimator


class F1TenthEulerStateEstimator(StateEstimator):
    """
    State estimator for the F1Tenth using a Moving Average Filter.
    """

    def __init__(self):
        super(F1TenthEulerStateEstimator, self).__init__()

        # Fetch additional parameters
        self.decay_rate = rospy.get_param("~decay_rate", 0.75)
        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_f1tenth/realm_f1tenth"
        )

        # Set a timer to poll the parameters
        self.poll_timer = rospy.Timer(
            rospy.Duration(1.0), self.poll_parameters_callback
        )

        # Initialize the Euler variables
        self.state = np.zeros((4, 1))  # [x, y, theta, v]
        self.last_pose = np.zeros((3, 1))  # [x, y, theta]

        # Set up subscribers
        self.last_position_msg = None
        self.position_sub = rospy.Subscriber(
            self.position_topic, TransformStamped, self.position_callback
        )

        # Publisher for PoseStamped messages
        self.estimate_pub = rospy.Publisher(
            f"{rospy.get_name()}/estimate", F1TenthState, queue_size=10
        )

    def poll_parameters_callback(self, _):
        """Poll the parameters from the parameter server."""
        new_decay_rate = rospy.get_param("~decay_rate", 0.75)

        if new_decay_rate != self.decay_rate:
            self.decay_rate = new_decay_rate
            rospy.loginfo(f"Updated decay rate to {self.decay_rate}")

    def reset_state(self, msg=None):
        """Reset the state of the Euler estimation."""
        self.state = np.zeros((4, 1))  # Reset state to zeros
        self.last_pose = np.zeros((3, 1))  # [x, y, theta]

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

            measured_pose = np.array([x, y, theta]).reshape(-1, 1)

            # Compare the current pose with the previous pose to estimate the velocity
            delta_position = measured_pose[:2] - self.last_pose[:2]
            distance_travelled = np.linalg.norm(delta_position)
            v_est = distance_travelled / self.dt

            measured_state = np.array([x, y, theta, v_est]).reshape(-1, 1)
 
            # Update the filter state
            self.state = measured_state
            self.last_pose = measured_pose

            self.last_position_msg = None

        # Publish the new state estimate
        msg = F1TenthState()
        msg.x = self.state[0, 0]
        msg.y = self.state[1, 0]
        msg.theta = self.state[2, 0]
        msg.speed = self.state[3, 0]
        self.estimate_pub.publish(msg)


if __name__ == "__main__":
    try:
        euler_node = F1TenthEulerStateEstimator()
        euler_node.run()
    except rospy.ROSInterruptException:
        pass
