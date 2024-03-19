#!/usr/bin/env python3
"""Simulate a turtlebot as a ROS node."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Twist


class TurtlebotSimulator:
    """Implement a simple simulator for a turtlebot with vicon (no other sensors)."""

    def __init__(self):
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("turtlebot_simulator")
        self.control_topic = rospy.get_param("~control_topic", "/cmd_vel")
        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_turtle0/realm_turtle0"
        )
        self.frame_name = rospy.get_param("~frame_name", "realm_turtle0")

        # Initialize the turtlebot state
        self.state = np.array([0.0, 0.0, 0.0])
        self.command = np.array([0.0, 0.0])

        # Set the simulation rate
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.rate = rospy.Rate(self.rate_hz)

        # Subscribe to cmd_vel
        self.cmd_vel_sub = rospy.Subscriber(
            self.control_topic, Twist, self.cmd_vel_callback
        )

        # Publish the transform of the turtlebot
        self.tf_pub = rospy.Publisher(
            self.position_topic, TransformStamped, queue_size=10
        )

    def cmd_vel_callback(self, msg):
        """Update the saved command."""
        self.command = np.array([msg.linear.x, msg.angular.z])

    def run(self):
        """Run the simulation."""
        while not rospy.is_shutdown():
            # Unpack the command then zero it out
            v, w = self.command
            self.command = np.array([0.0, 0.0])

            # Update the state
            theta = self.state[2]
            dq_dt = np.array([v * np.cos(theta), v * np.sin(theta), w])
            self.state += dq_dt / self.rate_hz

            # Publish the transform
            tf = TransformStamped()
            tf.header.stamp = rospy.Time.now()
            tf.header.frame_id = "world"
            tf.child_frame_id = self.frame_name
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
        sim_node = TurtlebotSimulator()
        sim_node.run()
    except rospy.ROSInterruptException:
        pass
