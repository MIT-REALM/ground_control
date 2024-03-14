#!usr/bin/env python
"""Define control for turtlebot using LQR """
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from rgc_state_estimators.msg import TurtlebotState

# from rgc_control.policies.tracking.steering_policies import TurtlebotSteeringPolicy
from rgc_control.policies.common import TurtlebotAction
from rgc_control.policies.tracking.tracking_policies import (
    TimedPose2DObservation,
)
from rgc_control.policies.tro_experiment_policies import create_tro_turtlebot_policy
from rgc_control.robot_control import RobotControl


class TurtlebotControl(RobotControl):
    def __init__(self):
        super(TurtlebotControl, self).__init__()

        # Publish cmd velocity for state estimation
        self.control_pub = rospy.Publisher("/cmd", Twist, queue_size=1)
        self.control = TurtlebotAction(0.0, 0.0)

        # Subscribe to State estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", f"{rospy.get_name()}/estimate"
        )
        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, TurtlebotState, self.state_estimate_callback
        )

        # Instantiate control policy using Turtlebot steering policy and reference
        # trajectory. We need to wait until we get the first state estimate in order
        # to instantiate the control policy.
        while self.state is None:
            rospy.loginfo(
                "Waiting for state estimate to converge to instantiate control policy"
            )
            rospy.sleep(1.0)

        rospy.sleep(2.0)  # additional waiting for state to converge
        rospy.loginfo("State estimate has converged. Instantiating control policy.")

        self.control_policy = create_tro_turtlebot_policy(
            np.array([self.state.x, self.state.y]), self.eqx_filepath
        )

    def state_estimate_callback(self, msg):
        self.state = msg

    def reset_control(self, msg=None):
        """Reset the control."""
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
        if self.state is not None:
            # Make sure to normalize the time.
            t = (rospy.Time.now() - self.time_begin).to_sec() / self.T

            # Pack [x,y,theta,v] from state message and control policy into
            # TimedPose2DObservation instance.
            v = self.control.linear_velocity
            t = rospy.Time.now() - self.time_begin
            current_state = TimedPose2DObservation(
                x=self.state.x, y=self.state.y, theta=self.state.theta, v=v, t=t
            )
            self.control = self.control_policy.compute_action(current_state)

        msg = Twist()
        msg.linear.x = self.control.linear_velocity
        msg.angular.z = self.control.angular_velocity
        self.control_pub.publish(msg)


if __name__ == "__main__":
    try:
        control_node = TurtlebotControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
