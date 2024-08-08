#!/usr/bin/env python3
"""Define class for robot control """
import os
import cv2
import numpy as np
import rospy
import math
from cv_bridge import CvBridge
from f1tenth_msgs.msg import F1TenthDriveStamped
from rgc_state_estimators.msg import F1TenthState
from sensor_msgs.msg import Image

# from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy
from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.ral_experiment_policies import (
    RALF1tenthObservation,
    create_ral_f1tenth_policy,
)
from rgc_control.robot_control import RobotControl
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D


class F1TenthControl(RobotControl):
    def __init__(self):
        super(F1TenthControl, self).__init__()

        # Publish cmd:[steering angle,acceleration] for state estimation
        self.control_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_0",
            F1TenthDriveStamped,
            queue_size=1,
        )
        self.desired_speed = 0.0
        self.control = F1TenthAction(0.0, 0.0)
        self.e = 0.0
        self.theta_e = 0.0

        # Subscribe to state estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", f"{rospy.get_name()}/estimate"
        )
        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, F1TenthState, self.state_estimate_callback
        )

        self.traj_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"), 
            rospy.get_param("~trajectory/filename")
        )
        self.v_ref = rospy.get_param("~v_ref", 0.5)        
        self.reference_trajectory = SplineTrajectory2D(self.v_ref,self.traj_filepath)

        print(self.reference_trajectory.cx[0],self.reference_trajectory.cy[0],self.reference_trajectory.cyaw[0],self.reference_trajectory.cx[-1],self.reference_trajectory.cy[-1])
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

        self.control_policy = create_ral_f1tenth_policy(
            np.array([self.state.x, self.state.y, self.state.theta, self.state.speed,self.e,self.theta_e]),
            self.v_ref,
            self.traj_filepath,
        )

        self.goal_x = self.reference_trajectory.cx[-1]
        self.goal_y = self.reference_trajectory.cy[-1]
        self.goal_yaw = self.reference_trajectory.cyaw[-1]
        self.goal_dist = 0.3


    def state_estimate_callback(self, msg):
        self.state = msg
        dx = self.state.x - self.goal_x
        dy = self.state.y - self.goal_y
        if math.hypot(dx, dy) <= self.goal_dist:
            print("goal reached :)")
            self.reset_control(None)

    def reset_control(self, msg=None):
        """Reset the control to stop the experiment and publish the command."""
        self.control = F1TenthAction(0.0, 0.0)
        msg = F1TenthDriveStamped()
        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration
        self.control_pub.publish(msg)
        self.desired_speed = 0.0

    # def calc_nearest_index(self,state, cx, cy, cyaw):
    #     dx = [state.x - icx for icx in cx]
    #     dy = [state.y - icy for icy in cy]
    #     d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    #     mind = min(d)
    #     ind = d.index(mind)
    #     mind = np.sqrt(mind)
    #     dxl = cx[ind] - state.x
    #     dyl = cy[ind] - state.y
    #     angle = self.pi_2_pi(cyaw[ind] - np.arctan2(dyl, dxl))
    #     if angle < 0:
    #         mind *= -1
    #     return ind, mind
    # def pi_2_pi(self,angle):
    #     return (angle + np.pi) % (2 * np.pi) - np.pi

    # def set_e(self,state,cx,cy,cyaw):
    #     e = np.sqrt(np.power((cx-state.x),2) + np.power((cy-state.y),2))
    #     dxl = cx - state.x  
    #     dyl = cy - state.y
    #     angle = self.pi_2_pi(cyaw - np.arctan2(dyl, dxl))
    #     if angle < 0:
    #         e *= -1
    #     theta_e = self.pi_2_pi(cyaw-state.theta)
    #     return e,theta_e
    
    def update(self):
        """
        Update and publish the control.
        This function implements fand calls the control prediction and update steps.
        """
        if self.state is not None:
            # Pack [x,y,theta,v] from state message into TimedPose2DObservation instance
            # Make sure to normalize the time
            t = (rospy.Time.now() - self.time_begin).to_sec() / self.T 

            #Calculate the nearest point on the curvature to steer towards
            # ind, _ = self.calc_nearest_index(self.state,self.reference_trajectory.cx,self.reference_trajectory.cy,self.reference_trajectory.cyaw)
            
            current_state = RALF1tenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                e = self.e,
                theta_e = self.theta_e,
                t=ind,
            )

            self.control, self.e, self.theta_e, ind = self.control_policy.compute_action(current_state)
            # self.e,self.theta_e = self.set_e(self.state,self.reference_trajectory.cx[ind],self.reference_trajectory.cy[ind],self.reference_trajectory.cyaw[ind])
            
            # # Stop if the experiment is over
            # if t >= 10.0:
            #     self.control = F1TenthAction(0.0, 0.0)

            print("state:")
            print(self.state)
            print("control:")
            print(self.control)
            print("goal:")
            print(ind, self.reference_trajectory.cx[ind], self.reference_trajectory.cy[ind])

        elif self.state is None:
            rospy.loginfo("No state estimate available!")

        msg = F1TenthDriveStamped()
        #msg.drive.mode = 1
        msg.drive.mode=0
        if self.control.steering_angle>=np.pi/4:
            self.control.steering_angle=np.pi/4
        elif self.control.steering_angle<=-np.pi/4:
            self.control.steering_angle = -np.pi/4

        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration

        # Control speed rather than acceleration directly
        self.desired_speed += self.dt * self.control.acceleration
        if self.desired_speed > 0.5:
            self.desired_speed = 0.5
        msg.drive.speed = self.desired_speed
        # print(msg.drive.speed,msg.drive.steering_angle)
        self.control_pub.publish(msg)


if __name__ == "__main__":
    try:
        control_node = F1TenthControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
