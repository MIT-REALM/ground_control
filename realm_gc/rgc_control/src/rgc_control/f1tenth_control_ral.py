#!/usr/bin/env python3
"""Define class for robot control """
import os
import numpy as np
import rospy
import math
from f1tenth_msgs.msg import F1TenthDriveStamped
from rgc_state_estimators.msg import F1TenthState
from sensor_msgs.msg import Image
import pickle

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

        self.filename = rospy.get_param("~trajectory/filename")

        self.traj_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"), 
            self.filename
        )
        self.v_ref = 3.53#rospy.get_param("~v_ref",5.0) 
        self.x_offset = rospy.get_param("~x_offset")
        self.y_offset = rospy.get_param("~y_offset")
        self.scale = rospy.get_param("~scale")       
        self.reference_trajectory = SplineTrajectory2D(self.v_ref,self.traj_filepath,
            self.scale, self.x_offset, self.y_offset)

        # cx = self.reference_trajectory.cx
        # cy = self.reference_trajectory.cy

        # self.reference_trajectory.cx = cy
        # self.reference_trajectory.cy = cx

        self.goal_x = self.reference_trajectory.cx[-1]
        self.goal_y = self.reference_trajectory.cy[-1]
        self.goal_yaw = self.reference_trajectory.cyaw[-1]
        self.goal_dist = 0.1
        self.goal_reached = False

        self.target_ind = 0

        self.data = {}
        self.data['v_ref'] = self.v_ref
        self.data['states'] = {}
        self.data['states']['x'] = []
        self.data['states']['y'] = []
        self.data['states']['theta'] = []
        self.data['states']['v'] = []
        self.data['states']['e'] = []
        self.data['states']['theta_e'] = []
        self.data['states']['target_ind'] = []
        self.data['control'] = {}
        self.data['control']['accel'] = []
        self.data['control']['angle'] = []

        self.data['filename'] = self.traj_filepath

        # Subscribe to state estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", f"{rospy.get_name()}/estimate"
        )
        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, F1TenthState, self.state_estimate_callback
        )


        #print("reference_trajectory start: ", 
        #    self.reference_trajectory.cx[0],self.reference_trajectory.cy[0],self.reference_trajectory.cyaw[0])
        #print("reference_trajectory end:   ",
        #    self.reference_trajectory.cx[-1],self.reference_trajectory.cy[-1])

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
            self.scale,
            self.x_offset,
            self.y_offset,
            self.v_ref,
            self.traj_filepath,
        )
        
    def state_estimate_callback(self, msg):
        self.state = msg
        # print("state msg:", msg)

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
            #print("Before",self.e)
            current_state = RALF1tenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                e = self.e,
                theta_e = self.theta_e,
                t=self.target_ind,
            )
            #print("State:",self.state.x,self.state.y,"Error:",(self.e),"time:",t)
            dx = self.state.x - self.goal_x
            dy = self.state.y - self.goal_y
            if math.hypot(dx, dy) <= self.goal_dist:
                # if not self.goal_reached:
                #     file = open(f"data_{self.filename.split('.')[0]}_{self.v_ref}.pkl", "wb")
                #     pickle.dump(self.data, file)
                #     file.close()
                #if self.goal_reached:
                print("goal reached :)")
                self.goal_reached = True
                self.state.speed=0
                self.reset_control()
                self.ctrl_c=True
            else:
                self.data['states']['x'].append(self.state.x)
                self.data['states']['y'].append(self.state.y)
                self.data['states']['theta'].append(self.state.theta)
                self.data['states']['v'].append(self.state.speed)
                self.data['states']['e'].append(self.e)
                self.data['states']['theta_e'].append(self.theta_e)
                self.data['states']['target_ind'].append(self.target_ind)
                
                #print("Before-1",self.e)
                #print("Current State:",current_state)
                self.control, self.e, self.theta_e, self.target_ind = self.control_policy.compute_action(current_state)
                #print("After",self.e)

                #print(self.target_ind, self.reference_trajectory.cx[self.target_ind], self.reference_trajectory.cy[self.target_ind], self.reference_trajectory.cyaw[self.target_ind])

                self.data['control']['accel'].append(self.control.acceleration)
                self.data['control']['angle'].append(self.control.steering_angle)

            file = open(f"/catkin_ws/src/realm_gc/rgc_control/saved_policies/data/data_{self.filename.split('.')[0]}.pkl", "wb")
            #print(f" stored at: ", f"/catkin_ws/src/realm_gc/data_{self.filename.split('.')[0]}_{self.v_ref}.pkl")
            pickle.dump(self.data, file)
            file.close()
            # self.e,self.theta_e = self.set_e(self.state,self.reference_trajectory.cx[ind],self.reference_trajectory.cy[ind],self.reference_trajectory.cyaw[ind])
            
            # # Stop if the experiment is over
            # if t >= 10.0:
            #     self.control = F1TenthAction(0.0, 0.0)

            # print("state:")
            # print(self.state)
            # print("control:")
            # print(self.control)
            # print("goal:")
            # ind = self.target_ind
            # print(ind, self.reference_trajectory.cx[ind], self.reference_trajectory.cy[ind])

        elif self.state is None:
            rospy.loginfo("No state estimate available!")

        msg = F1TenthDriveStamped()
        #msg.drive.mode = 1
        msg.drive.mode=0
        if not self.goal_reached:
            # if self.control.steering_angle>=np.pi/8:
            #     self.control.steering_angle=np.pi/8
            # elif self.control.steering_angle<=-np.pi/8:
            #     self.control.steering_angle = -np.pi/8
            self.control.steering_angle = min( np.pi/3, self.control.steering_angle)
            self.control.steering_angle = max(-np.pi/3, self.control.steering_angle)

            msg.drive.steering_angle = self.control.steering_angle
            msg.drive.acceleration = self.control.acceleration

            # Control speed rather than acceleration directly
            self.desired_speed += self.dt * self.control.acceleration
            self.desired_speed = min(self.desired_speed, self.v_ref)
            self.desired_speed = min( self.v_ref, self.desired_speed)
            self.desired_speed = max(-self.v_ref, self.desired_speed)
            print("Speed:",self.state.speed, "Ref:",self.v_ref, "Acc:",self.control.acceleration)
            msg.drive.speed = self.desired_speed
            #print(msg.drive.speed,msg.drive.acceleration,msg.drive.steering_angle)
            self.control_pub.publish(msg)



if __name__ == "__main__":
    try:
        control_node = F1TenthControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass
