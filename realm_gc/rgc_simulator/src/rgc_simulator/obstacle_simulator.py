#!/usr/bin/env python3
"""Simulate a generic obstacle as a ROS node."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Twist


class ObstacleSimulator:
    """Implement a simple simulator for a generic obstacle with vicon (no other sensors)."""

    def __init__(self):
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("obstacle_simulator")

        self.x_pos = rospy.get_param("~y_pos", 0.0)
        self.y_pos = rospy.get_param("~x_pos", 0.0)
        self.theta = rospy.get_param("~angle", 0.0)

        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_obstacle0/realm_obstacle0"
        )
        self.frame_name = rospy.get_param("~frame_name", "realm_obstacle0")

        print(self.x_pos, self.y_pos, self.angle, self.position_topic, self.frame_name)

        # Initialize the obstacle state
        self.state = np.array([self.x_pos, self.y_pos, self.theta])
        self.command = np.array([0.0, 0.0])

        # Set the simulation rate
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.rate = rospy.Rate(self.rate_hz)

        # Publish the transform of the obstacle
        self.tf_pub = rospy.Publisher(
            self.position_topic, TransformStamped, queue_size=10
        )

    def run(self):
        """Run the simulation."""
        while not rospy.is_shutdown():
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
        sim_node = ObstacleSimulator()
        sim_node.run()
    except rospy.ROSInterruptException:
        pass
