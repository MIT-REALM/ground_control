#!/usr/bin/env python3
"""Define class for robot control """
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from f1tenth_msgs.msg import F1TenthDriveStamped
from rgc_state_estimators.msg import F1TenthState
from sensor_msgs.msg import Image

# from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy
from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.tro_experiment_policies import (
    TROF1TenthObservation,
    create_tro_f1tenth_policy,
)
from rgc_control.robot_control import RobotControl


class F1TenthControl(RobotControl):
    def __init__(self):
        super(F1TenthControl, self).__init__()

        # Publish cmd:[steering angle,acceleration] for state estimation
        self.control_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_0",
            F1TenthDriveStamped,
            queue_size=1,
        )
        self.control = F1TenthAction(0.0, 0.0)

        # Subscribe to state estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", f"{rospy.get_name()}/estimate"
        )
        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, F1TenthState, self.state_estimate_callback
        )

        # Subscribe to depth images from ros param
        self.depth_image = None
        image_width = 16
        aspect = 4.0 / 3.0
        self.image_shape = (image_width, int(image_width / aspect))
        self.depth_image = None
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            rospy.get_param("~depth_image_topic", "/camera/depth/image_rect_raw"),
            Image,
            self.depth_image_callback,
        )

        # Get MLP eqx filepath from rosparam supplied by roslaunch
        self.mlp_eqx = os.path.join(
            rospy.get_param("~mlp/base_path"), rospy.get_param("~mlp/filename")
        )

        # Instantiate control policy using F1Tenth steering policy and reference
        # trajectory. We need to wait until we get the first state estimate in order
        # to instantiate the control policy.
        while self.state is None:
            rospy.loginfo(
                "Waiting for state estimate to converge to instantiate control policy"
            )
            rospy.sleep(1.0)

        rospy.sleep(2.0)  # additional waiting for state to converge
        rospy.loginfo("State estimate has converged. Instantiating control policy.")

        self.control_policy = create_tro_f1tenth_policy(
            np.array([self.state.x, self.state.y, self.state.theta, self.state.speed]),
            self.eqx_filepath,
            self.mlp_eqx,
        )

    def state_estimate_callback(self, msg):
        self.state = msg

    def depth_image_callback(self, msg):
        original_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_image = cv2.resize(
            original_image, self.image_shape, interpolation=cv2.INTER_AREA
        )

    def reset_control(self, msg=None):
        """Reset the control to stop the experiment and publish the command."""
        self.control = F1TenthAction(0.0, 0.0)
        msg = F1TenthDriveStamped()
        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration
        self.control_pub.publish(msg)

    def update(self):
        """
        Update and publish the control.
        This function implements and calls the control prediction and update steps.
        """
        if self.state is not None and self.img is not None:
            # Pack [x,y,theta,v] from state message into TimedPose2DObservation instance
            # Make sure to normalize the time
            t = (rospy.Time.now() - self.time_begin).to_sec() / self.T
            current_state = TROF1TenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                t=t,
                depth_image=self.depth_image,
            )
            self.control = self.control_policy.compute_action(current_state)

        msg = F1TenthDriveStamped()
        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration
        self.control_pub.publish(msg)


if __name__ == "__main__":
    try:
        control_node = F1TenthControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
