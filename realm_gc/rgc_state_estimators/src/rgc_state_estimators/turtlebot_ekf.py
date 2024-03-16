#!/usr/bin/env python3
"""Define a state estimator for the turtlebot using an Extended Kalman Filter."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Twist
from transforms3d.euler import quat2euler

from rgc_state_estimators.msg import TurtlebotState, TurtlebotStateCovariance
from rgc_state_estimators.state_estimator import StateEstimator


class TurtlebotEKFStateEstimator(StateEstimator):
    """
    State estimator for the Turtlebot using an Extended Kalman Filter.
    """

    def __init__(self):
        super(TurtlebotEKFStateEstimator, self).__init__()

        # Fetch additional parameters
        self.obs_noise_cov = rospy.get_param("~obs_noise_cov", 0.1)
        self.process_noise_cov = rospy.get_param("~process_noise_cov", 0.1)
        self.control_topic = rospy.get_param("~control_topic", "cmd_vel")
        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_turtle0/realm_turtle0"
        )

        # Set a timer to poll the parameters
        self.poll_timer = rospy.Timer(
            rospy.Duration(1.0), self.poll_parameters_callback
        )

        # Initialize the EKF variables
        self.state = np.zeros((3, 1))  # [x, y, theta]
        self.covariance = np.eye(3)  # Initial covariance matrix

        # Set up subscribers
        self.last_control_msg = None
        self.control_sub = rospy.Subscriber(
            self.control_topic, Twist, self.control_callback
        )
        self.last_position_msg = None
        self.position_sub = rospy.Subscriber(
            self.position_topic, TransformStamped, self.position_callback
        )

        # Publisher for PoseStamped messages
        self.estimate_pub = rospy.Publisher(
            f"{rospy.get_name()}/estimate", TurtlebotState, queue_size=10
        )
        self.covariance_pub = rospy.Publisher(
            f"{rospy.get_name()}/estimate_covariance",
            TurtlebotStateCovariance,
            queue_size=10,
        )

    def poll_parameters_callback(self, _):
        """Poll the parameters from the parameter server."""
        new_obs_noise_cov = rospy.get_param("~obs_noise_cov", 0.1)
        new_process_noise_cov = rospy.get_param("~process_noise_cov", 0.1)

        if new_obs_noise_cov != self.obs_noise_cov:
            self.obs_noise_cov = new_obs_noise_cov
            rospy.loginfo(
                f"Updated observation noise covariance to {self.obs_noise_cov}"
            )

        if new_process_noise_cov != self.process_noise_cov:
            self.process_noise_cov = new_process_noise_cov
            rospy.loginfo(
                f"Updated process noise covariance to {self.process_noise_cov}"
            )

    def reset_state(self, msg=None):
        """Reset the state of the EKF."""
        self.state = np.zeros((3, 1))  # Reset state to zeros
        self.covariance = np.eye(3)  # Reset covariance to identity

    def unicycle_model(self, state, v, w):
        """
        Update the state using the unicycle model.
        Args:
            state (np.ndarray): The current state [x, y, theta]
            v (float): The velocity command
            w (float): The angular velocity command
        Returns:
            np.ndarray: The updated state
        """
        # Extract the state variables
        x, y, theta = state

        # Update the state using the unicycle model
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += w * self.dt

        return np.array([x, y, theta]).reshape(-1, 1)

    def get_AB(self, state, v, w):
        """
        Compute the linearized dynamics matrices.

        Args:
            state (np.ndarray): The current state [x, y, theta]
            v (float): The velocity command
            w (float): The angular velocity command
        """
        # Extract the state variables
        _, _, theta = state

        # Compute the linearized dynamics matrices
        A = np.eye(3)
        A[0, 2] = -v * np.sin(theta) * self.dt
        A[1, 2] = v * np.cos(theta) * self.dt

        B = np.zeros((3, 2))
        B[0, 0] = np.cos(theta) * self.dt
        B[1, 0] = np.sin(theta) * self.dt
        B[2, 1] = self.dt

        return A, B

    def control_callback(self, msg):
        """
        Update the state based on control inputs.
        """
        self.last_control_msg = msg

    def position_callback(self, msg):
        """
        Update the state based on new position measurements.
        Placeholder function - implement measurement update logic here.
        """
        self.last_position_msg = msg

    def update(self):
        """
        Update the filter state and publish the new state estimate.
        This function should implement or call the EKF prediction and update steps.
        """
        if self.last_control_msg is not None:
            # Extract control inputs from the message
            v = self.last_control_msg.linear.x
            w = self.last_control_msg.angular.z

            # Update the state based on the control inputs
            self.state = self.unicycle_model(self.state, v, w)

            # Update the covariance based on the linearized dynamics
            A, _ = self.get_AB(self.state, v, w)
            Q = np.eye(3) * self.process_noise_cov
            self.covariance = A @ self.covariance @ A.T + Q

            self.last_control_msg = None

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

            # The measurement matrix H is the identity matrix
            H = np.eye(3)

            # Do the measurement update step of the EKF
            R = np.eye(3) * self.obs_noise_cov
            error = np.array([x, y, theta]).reshape(-1, 1) - self.state
            S = H @ self.covariance @ H.T + R
            K = self.covariance @ H.T @ np.linalg.inv(S)
            self.state += K @ error
            self.covariance = (np.eye(3) - K @ H) @ self.covariance

            self.last_position_msg = None

        # Publish the new state estimate
        msg = TurtlebotState()
        msg.x = self.state[0, 0]
        msg.y = self.state[1, 0]
        msg.theta = self.state[2, 0]
        self.estimate_pub.publish(msg)

        # Publish the new covariance estimate
        cov_msg = TurtlebotStateCovariance()
        cov_msg.covariance = self.covariance.ravel().tolist()
        self.covariance_pub.publish(cov_msg)


if __name__ == "__main__":
    try:
        ekf_node = TurtlebotEKFStateEstimator()
        ekf_node.run()
    except rospy.ROSInterruptException:
        pass
