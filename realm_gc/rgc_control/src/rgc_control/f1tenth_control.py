#!usr/bin/env python
"""Define class for robot control """
import numpy as np
import rospy
from f1tenth_msgs.msg import F1TenthDriveStamped
from rgc_state_estimators.msg import F1TenthState

# from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy
from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.tracking.tracking_policies import (
    TimedPose2DObservation,
)
from rgc_control.policies.tro_experiment_policies import create_tro_f1tenth_policy
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

        # Get MLP eqx filepath from rosparam supplied by roslaunch
        self.mlp_eqx = rospy.get_param("~mlp/base_path") + rospy.get_param(
            "~mlp/filename"
        )

        # Instantiate control policy using F1Tenth steering policy and reference
        # trajectory
        self.control_policy = create_tro_f1tenth_policy(
            np.zeros((4, 1)), self.eqx_filepath, self.mlp_eqx
        )

        # Start time
        self.time_begin = rospy.Time.now()

    def shutdownhook(self):
        self.ctrl_c = True

    def state_estimate_callback(self, msg):
        self.state = msg

    def reset_control(self, msg=None):
        """Reset the control."""
        self.control = F1TenthAction(0.0, 0.0)

    def update(self):
        """
        Update and publish the control.
        This function implements and calls the control prediction and update steps.
        """
        if self.state is not None:
            # Pack [x,y,theta,v] from state message into TimedPose2DObservation instance
            t = rospy.Time.now() - self.time_begin
            current_state = TimedPose2DObservation(
                self.state.x, self.state.y, self.state.theta, self.state.speed, t
            )
            self.control = self.control_policy.compute_action(current_state)

        msg = F1TenthDriveStamped()
        msg.steering_angle = self.control.steering_angle
        msg.acceleration = self.control.acceleration
        self.control_pub.publish(msg)


if __name__ == "__main__":
    try:
        control_node = F1TenthControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
