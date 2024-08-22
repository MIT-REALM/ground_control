#!/usr/bin/env python3
"""Define class for robot control """
import os

import cv2
import numpy as np
import rospy
# from cv_bridge import CvBridge
from f1tenth_msgs.msg import F1TenthDriveStamped
from rgc_state_estimators.msg import F1TenthState
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy
from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.icra_experiment_policies import (
    ICRAF1TenthObservation,
    # create_icra_f1tenth_policy,
)

from rgc_control.policies.tracking.trajectory import SplineTrajectory2D

from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy, SteeringObservation, Pose2DObservation

from rgc_control.robot_control import RobotControl
from rgc_control.policies.gcbf_policy import GCBF_policy

from rgc_control.policies.ral_experiment_policies import (
    create_ral_f1tenth_policy,RALF1tenthObservation
)

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

        # Subscribe to state estimation topic from ros param
        self.state = None
        self.state_estimate_topic = rospy.get_param(
            "~state_estimate_topic", f"{rospy.get_name()}/estimate"
        )
        self.sim_state = None
        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_f1tenth/realm_f1tenth"
        )

        self.sim_position_topic = rospy.get_param(
            "~position_topic_sim", "/vicon/realm_f1tenth/realm_f1tenth_sim"
        )

        self.position_sub = rospy.Subscriber(
            self.position_topic, TransformStamped, self.pos_callback
        )

        self.sim_position_sub = rospy.Subscriber(
            self.sim_position_topic, F1TenthState, self.sim_pos_callback
        )

        self.estimate_sub = rospy.Subscriber(
            self.state_estimate_topic, F1TenthState, self.state_estimate_callback
        )
        self.obs1 = None
        self.obs_state_topic = rospy.get_param(
            "~position_topic1", "/vicon/realm_f1tenth/realm_obs")
        
        self.obs_state_sub = rospy.Subscriber(
            self.obs_state_topic, TransformStamped, self.obs_state_callback
        )
        
        self.obs2 = None
        self.obs_state_topic = rospy.get_param("~position_topic2",
            "/vicon/realm_f1tenth/realm_obs2")
        
        self.obs_state_sub = rospy.Subscriber(
            self.obs_state_topic, TransformStamped, self.obs_state_callback2
        )


        # Instantiate control policy using F1Tenth steering policy and reference
        # trajectory. We need to wait until we get the first state estimate in order
        # to instantiate the control policy.
        while self.state is None:
            rospy.loginfo(
                "Waiting for state estimate to converge to instantiate control policy"
            )
            rospy.sleep(1.0)

        while self.obs1 is None:
            rospy.loginfo(
                "Waiting for obs1 state estimate to converge to instantiate control policy"
            )
            rospy.sleep(1.0)
        
        while self.obs2 is None:
            rospy.loginfo(
                "Waiting for obs2 state estimate to converge to instantiate control policy"
            )
            rospy.sleep(1.0)

        # rospy.sleep(2.0)  # additional waiting for state to converge
        rospy.loginfo("State estimate has converged. Instantiating control policy.")

        # self.control_policy = create_icra_f1tenth_policy(
        #     np.array([self.state.x, self.state.y, self.state.theta, self.state.speed]),
        #     self.eqx_filepath,
        # )

        self.traj_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"), 
            rospy.get_param("~trajectory/filename")
        )
        self.v_ref = rospy.get_param("~v_ref", 0.5)        
        self.reference_trajectory = SplineTrajectory2D(self.v_ref,self.traj_filepath)

        self.goal_x = self.reference_trajectory.cx[-1]
        self.goal_y = self.reference_trajectory.cy[-1]
        self.goal_yaw = self.reference_trajectory.cyaw[-1]
        self.e = 0.0
        self.theta_e = 0.0
        self.target_ind = 0

        goal = np.array([self.goal_x, self.goal_y, self.goal_yaw, 0.0])
        self.goal = goal
        car_pos = np.array([self.reference_trajectory.cx[0], self.reference_trajectory.cy[0], self.reference_trajectory.cyaw[0], 0.0])

        obs_pos = np.array([[self.obs1[0], self.obs1[1]], [self.obs2[0], self.obs2[1]]])
        self.control_policy = GCBF_policy(
            min_distance=1.0,
            car_pos=car_pos,
            car_goal=goal,
            obs_pos=obs_pos,
            num_obs=1,
            mov_obs=2,
            model_path='/catkin_ws/src/realm_gc/rgc_control/src/gcbfplus/seed1_20240719162242/'
        )

        self.reference_control = F1TenthSteeringPolicy(equilibrium_state=car_pos + np.array([0.1, -0.2, -0.2, 0.1]), axle_length=0.28, dt=0.03)

        self.current_pose = None
        self.goal_pose= Pose2DObservation(
            goal[0],
            goal[1],
            goal[2],
            goal[3]
        )
        self.obs= None

        self.control_policy_ral = create_ral_f1tenth_policy(
            np.array([self.state.x, self.state.y, self.state.theta, self.state.speed,self.e,self.theta_e]),
            self.v_ref,
            self.traj_filepath,
        )

        self.new_traj = None
        self.first_step = True
        self.obs_pos_past = obs_pos

    def pos_callback(self, msg):
        self.actual_state = np.array([msg.transform.translation.x, msg.transform.translation.y, 0.0, 0.0])
    
    def sim_pos_callback(self, msg):
        self.sim_state = msg
        # print('actual_state:', msg.transform.translation.x, msg.transform.translation
    def obs_state_callback(self, msg):
        self.obs1 = np.array([msg.transform.translation.x, msg.transform.translation.y, 0.0, 0.0])
        # print('obs_state:', msg.transform.translation.x, msg.transform.translation.y)

    def obs_state_callback2(self, msg):
        self.obs2 = np.array([msg.transform.translation.x, msg.transform.translation.y, 0.0, 0.0])
        # print('obs_state:', msg.transform.translation.x, msg.transform.translation.y)
    
    def state_estimate_callback(self, msg):
        self.state = msg

    def reset_control(self, msg=None):
        """Reset the control to stop the experiment and publish the command."""
        self.control = F1TenthAction(0.0, 0.0)
        msg = F1TenthDriveStamped()
        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration
        self.control_pub.publish(msg)
        self.desired_speed = 0.0

    def update(self):
        """
        Update and publish the control.
        This function implements and calls the control prediction and update steps.
        """
        self.state = self.sim_state
        if self.state is not None:
            # Pack [x,y,theta,v] from state message into TimedPose2DObservation instance
            # Make sure to normalize the time
            t = (rospy.Time.now() - self.time_begin).to_sec() / self.T
            # print('self.state:', self.state)
            current_state = ICRAF1TenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                t=t,
            )

            # current_state = ICRAF1TenthObservation(
            #     x=self.state.transform.translation.x,
            #     y=self.state.transform.translation.y,
            #     theta=self.state.transform.rotation.w,
            #     v=self.state.transform.translation.z,
            #     t=t,
            # )
            # print(current_state.x)
            # print(current_state.y)

            current_state_timed = RALF1tenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                e = self.e,
                theta_e = self.theta_e,
                t=self.target_ind,
            )

            
            # spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath, self.new_traj)
            # _, min_dist = spline_traj.calc_nearest_index(self.state)
            # print('min_dist:', min_dist)

            # if min_dist > 0.2:
            #     traj = {}
            #     c = np.linspace(0, 1, 200)
            #     x_ref = self.state.x * (1 - c) + self.goal[0] * c
            #     y_ref = self.state.y * (1 - c) + self.goal[1] * c
            #     traj['X'] = x_ref 
            #     traj['Y'] = y_ref

            #     self.new_traj = traj
            #     self.control_policy_ral = create_ral_f1tenth_policy(
            #         np.array([self.state.x, self.state.y, self.state.theta, self.state.speed,self.e,self.theta_e]),
            #         self.v_ref,
            #         self.traj_filepath,
            #         traj=traj
            #     )

            #     current_state_timed = RALF1tenthObservation(
            #         x=self.state.x,
            #         y=self.state.y,
            #         theta=self.state.theta,
            #         v=self.state.speed,
            #         e = self.e,
            #         theta_e = self.theta_e,
            #         t=0,
            #     )

            reference_control, self.e, self.theta_e, self.target_ind = self.control_policy_ral.compute_action(current_state_timed)
            
            
            # self.current_pose = Pose2DObservation(
            #     self.state.x,
            #     self.state.y,
            #     self.state.theta, 
            #     self.state.speed,
            # )
            # self.obs= SteeringObservation(
            #     pose=self.current_pose,
            #     goal = self.goal_pose
            # )
            # current_state_np = np.array([self.state.x, self.state.y, self.state.theta, self.state.speed]) 

            # self.reference_control.set_eq(current_state_np)
            
            # reference_control = self.reference_control.compute_action(self.obs)

            obs_pos = np.array([[self.obs1[0], self.obs1[1]], [self.obs2[0], self.obs2[1]]])
            obs_pos_old = self.obs_pos_past
            obs_pos_new = obs_pos
            obs_vel = (obs_pos_new - obs_pos_old) / self.dt
            # if self.first_step:
            control_gcbf, next_state = self.control_policy.compute_action(current_state, reference_control, obs=obs_pos, mov_obs_vel=obs_vel)


            traj = {}
            c = np.linspace(0, 1, 20)
            # x_ref1 = self.state.x * (1 - c) + next_state[0] * c
            # y_ref1 = self.state.y * (1 - c) + next_state[1] * c

            # x_ref2 = next_state[0] * (1 - c) + self.goal[0] * c
            # y_ref2 = next_state[1] * (1 - c) + self.goal[1] * c

            # x_ref = np.concatenate((x_ref1, x_ref2))
            # y_ref = np.concatenate((y_ref1, y_ref2))
            
            x_ref = self.state.x * (1 - c) + self.goal[0] * c
            y_ref = self.state.y * (1 - c) + self.goal[1] * c
            # x_ref[2] = next_state[0]
            # y_ref[2] = next_state[1]
            
            traj['X'] = x_ref 
            traj['Y'] = y_ref 
            # if np.isnan(self.state.speed):
            #     self.state.speed = 0.0

            # self.new_traj = traj  
            next_state_pose = self.state
            next_state_pose.x = next_state[0]
            next_state_pose.y = next_state[1]
            next_state_pose.theta = next_state[2]
            next_state_pose.speed = next_state[3]

            spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath, traj)
            _, min_dist = spline_traj.calc_nearest_index(next_state_pose)
            print('min_dist:', min_dist)

            if min_dist < 0.01:
                self.control = reference_control
            else:
                x_ref[2] = next_state[0]
                y_ref[2] = next_state[1]
                
                traj['X'] = x_ref 
                traj['Y'] = y_ref 
                self.control_policy_ral = create_ral_f1tenth_policy(
                    np.array([self.state.x, self.state.y, self.state.theta, self.state.speed,self.e,self.theta_e]),
                    self.v_ref,
                    self.traj_filepath,
                    traj=traj
                )

                current_state_timed = RALF1tenthObservation(
                    x=self.state.x,
                    y=self.state.y,
                    theta=self.state.theta,
                    v=self.state.speed,
                    e = self.e,
                    theta_e = self.theta_e,
                    t=0,
                )

                control_steer, self.e, self.theta_e, self.target_ind = self.control_policy_ral.compute_action(current_state_timed)

                # spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath, traj)
                # _, min_dist = spline_traj.calc_nearest_index(self.state)
                # print('min_dist:', min_dist)

                # if min_dist > 0.2:
                #     self.control = control_steer
                # else:
                #     self.control = control_gcbf
                self.control = control_steer

            if np.isnan(self.control.steering_angle):
                if np.isnan(control_gcbf.steering_angle):
                    self.control.steering_angle = 0.0
                else:
                    self.control.steering_angle = control_gcbf.steering_angle
            
            if np.isnan(self.control.acceleration):
                if np.isnan(control_gcbf.acceleration):
                    self.control.acceleration = 0.0
                else:
                    self.control.acceleration = control_gcbf.acceleration

            if t >= 10.0:
                self.control = F1TenthAction(0.0, 0.0)

        elif self.state is None:
            rospy.loginfo("No state estimate available!")
    
        msg = F1TenthDriveStamped()
        msg.drive.mode = 1
        msg.drive.steering_angle = self.control.steering_angle
        msg.drive.acceleration = self.control.acceleration

        # Control speed rather than acceleration directly
        self.desired_speed += self.dt * self.control.acceleration
        
        if self.desired_speed > self.v_ref:
            self.desired_speed = self.v_ref
            msg.drive.acceleration = 0.0
        elif self.desired_speed < 0.0:
            self.desired_speed = -0.001
            msg.drive.acceleration = 0.0

        # msg.drive.mode = 0
        msg.drive.speed = self.desired_speed
    
        self.control_pub.publish(msg)
        print('control:', self.control.steering_angle, self.control.acceleration)
        print('speed:', self.desired_speed)
        dist_goal = np.sqrt((current_state.x - self.goal[0])**2 + (current_state.y - self.goal[1])**2)
        print('distance to goal:', dist_goal)
        if dist_goal < 0.1:
            print('Goal reached')
            self.reset_control()
            rospy.signal_shutdown('Goal reached')
        # rospy.sleep(0.01)
        # rospy.spin()

if __name__ == "__main__":
    try:
        control_node = F1TenthControl()
        control_node.run()
    except rospy.ROSInterruptException:
        pass