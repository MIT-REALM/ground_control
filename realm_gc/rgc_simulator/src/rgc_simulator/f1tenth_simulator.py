#!/usr/bin/env python3
"""Simulate f1tenth as a ROS node."""
import numpy as np
import rospy
from f1tenth_msgs.msg import F1TenthDriveStamped
from geometry_msgs.msg import TransformStamped


class F1TenthSimulator:
    """Implement a simple simulator for the f1tenth with vicon (no other sensors)."""

    def __init__(self):
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("f1tenth_simulator")
        self.axle_length = rospy.get_param("~axle_length", 0.28)
        self.control_topic = rospy.get_param(
            "~control_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
        )
        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_f1tenth/realm_f1tenth"
        )

        # Initialize the f1tenth state
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.command = np.array([0.0, 0.0])

        # Set the simulation rate
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.rate = rospy.Rate(self.rate_hz)

        # Subscribe to cmd_vel
        self.cmd_vel_sub = rospy.Subscriber(
            self.control_topic, F1TenthDriveStamped, self.cmd_callback
        )

        # Publish the transform of the f1tenth
        self.tf_pub = rospy.Publisher(
            self.position_topic, TransformStamped, queue_size=10
        )

    def cmd_callback(self, msg):
        """Update the saved command."""
        self.command = np.array([msg.drive.steering_angle, msg.drive.acceleration])

    def run(self):
        """Run the simulation."""
        while not rospy.is_shutdown():
            # Update the state
            x, y, theta, v = self.state
            delta, a = self.command
            dq_dt = np.array(
                [
                    v * np.cos(theta),
                    v * np.sin(theta),
                    (v / self.axle_length) * np.tan(delta),
                    a,
                ]
            )
            self.state += dq_dt / self.rate_hz

            # Publish the transform
            tf = TransformStamped()
            tf.header.stamp = rospy.Time.now()
            tf.header.frame_id = "world"
            tf.child_frame_id = "realm_f1tenth"
            tf.transform.translation.x = self.state[0]
            tf.transform.translation.y = self.state[1]
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = 0.0
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = np.sin(self.state[2] / 2)
            tf.transform.rotation.w = np.cos(self.state[2] / 2)
            self.tf_pub.publish(tf)

            # Sleep
            self.rate.sleep()


if __name__ == "__main__":
    try:
        sim_node = F1TenthSimulator()
        sim_node.run()
    except rospy.ROSInterruptException:
        pass
