#!/usr/bin/env python3
"""Define class for robot control """
import os

import cv2
import numpy as np
import rospy
# from cv_bridge import CvBridge
from f1tenth_msgs.msg import F1TenthDriveStamped, MultiArray
from rgc_state_estimators.msg import F1TenthState
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy
from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.icra_experiment_policies import (
    ICRAF1TenthObservation,
    # create_icra_f1tenth_policy,
)
# from std_msgs.msg import Float32MultiArray
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D

from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy, SteeringObservation, Pose2DObservation, F1TenthSpeedSteeringPolicy, SpeedSteeringObservation, G2CGoal2DObservation, G2CPose2DObservation

from rgc_control.robot_control import RobotControl
# from rgc_control.policies.gcbf_policy import GCBF_policy
from rgc_control.policies.cmarl_policy import CMARL_policy


from rgc_control.policies.ral_experiment_policies import (
    create_ral_f1tenth_policy,RALF1tenthObservation
)

from tf.transformations import euler_from_quaternion, quaternion_from_euler

import pytictoc

pytic = pytictoc.TicToc()

class F1TenthControl(RobotControl):
    def __init__(self):
        super(F1TenthControl, self).__init__()

        # Publish cmd:[steering angle,acceleration] for state estimation
        self.control_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/input/nav_0",
            F1TenthDriveStamped,
            queue_size=1,
        )

        self.traj_pub = rospy.Publisher(
            "/vesc/high_level/ackermann_cmd_mux/traj",
            MultiArray,
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
            "~position_topic1", "/vicon/realm_obs/realm_obs")
        
        self.obs_state_sub = rospy.Subscriber(
            self.obs_state_topic, TransformStamped, self.obs_state_callback
        )
        
        self.obs2 = None
        self.obs_state_topic = rospy.get_param("~position_topic2",
            "/vicon/realm_obs2/realm_obs2")
        
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

        print('state estimator converged!')

        print('state here: ', self.state)

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
        
        print('obs positions obtained: ',self.obs1, self.obs2)


        self.traj_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"), 
            rospy.get_param("~trajectory/filename")
        )
        # self.v_ref = rospy.get_param("~v_ref", 2.0)     
        self.v_ref = 0.7 
        
        self.goal_x = 0.0
        self.goal_y = 4.0
        self.goal_yaw = 0.0
        self.e = 0.0
        self.theta_e = 0.0
        self.target_ind = 0
        self.control_state = None
        goal = np.array([self.goal_x, self.goal_y, self.goal_yaw, 0.0])
        self.goal = goal
        self.accel_limit = 0.1
        # self.state.y = -2.0

        car_pos = np.array([self.state.x, self.state.y, self.state.theta, self.state.speed])

        obs_pos = np.array([[self.obs1[0], self.obs1[1]], [self.obs2[0], self.obs2[1]]])

        obs_center = obs_pos
        obs_r = 0.25
        theta = np.linspace(0, 2*np.pi, 10)
        circ = 1.5 * np.concatenate((np.cos(theta)[:, None], np.sin(theta)[:, None]), axis=1)
        
        obs1 = np.repeat(obs_center[0, :][:, None], 10, axis=1).T + circ
        obs2 = np.repeat(obs_center[1, :][:, None], 10, axis=1).T + circ
        obs = np.concatenate((obs1, obs2), axis=0)
        # print('obs shape: ', obs.shape)

        self.control_policy = CMARL_policy(
            min_distance=1.0,
            car_pos=car_pos,
            car_goal=goal,
            obs_pos=obs,
            num_obs=1,
            mov_obs=obs.shape[0],
            model_path='/catkin_ws/src/realm_gc/rgc_control/src/cmarl/logs/LidarF1TenthTarget/gcbfcrpo/seed0_926111104'
        )

        # c = np.linspace(0, 1, 10)
        # x = car_pos[0] * (1-c) + goal[0] * c
        # y = car_pos[1] * (1-c) + goal[1] * c
        # traj = {}
        # traj['X'] = x
        # traj['Y'] = y

        self.reference_trajectory = SplineTrajectory2D(self.v_ref,self.traj_filepath)
        print('init traj start: ', self.reference_trajectory.cx[0], self.reference_trajectory.cy[0])
        print('init traj end: ', self.reference_trajectory.cx[-1], self.reference_trajectory.cy[-1])
        # self.reference_control = F1TenthSteeringPolicy(equilibrium_state=car_pos + np.array([0.1, -0.2, -0.2, 0.1]), axle_length=0.28, dt=0.03)
        self.reference_control = F1TenthSpeedSteeringPolicy(trajectory=self.reference_trajectory, wheelbase=0.28, dt=self.dt, target_speed=self.v_ref)
        # ref_offset = np.array([0.1, -0.2, -0.2, 0.1])
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
        # self.actual_state = np.array([msg.transform.translation.x, msg.transform.translation.y, 0.0, 0.0])
        speed = 0.0
        theta = euler_from_quaternion([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
        self.actual_state = np.array([msg.transform.translation.x, msg.transform.translation.y, theta[-1], speed])
    
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
        msg.drive.speed = 0.0
        self.control_pub.publish(msg)
        self.desired_speed = 0.0

    def update(self):
        """
        Update and publish the control.
        This function implements and calls the control prediction and update steps.
        """
        
        ## use for sim not exps

        # if self.control_state is not None:
        #     self.state = F1TenthState()
        #     self.state.x = self.control_state[0]
        #     self.state.y = self.control_state[1]
        #     self.state.theta = self.control_state[2]
        #     self.state.speed = self.control_state[3]
            # self.state = self.control_state
        
        current_state = ICRAF1TenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                t=0,
            )
        
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


            current_state_timed = RALF1tenthObservation(
                x=self.state.x,
                y=self.state.y,
                theta=self.state.theta,
                v=self.state.speed,
                e = self.e,
                theta_e = self.theta_e,
                t=self.target_ind,
            )

            traj = {}
            c = np.linspace(0, 1, 10)
            
            x_ref = self.state.x * (1 - c) + self.goal[0] * c
            y_ref = self.state.y * (1 - c) + self.goal[1] * c
            
            traj['X'] = x_ref
            traj['Y'] = y_ref 
            
            # spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath, traj)
            # pytic.tic()
            spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath)  
            # print('first spline generation time: ', pytic.tocvalue())  
            traj = spline_traj
            ind, _ = spline_traj.calc_nearest_index(self.state)
            # closest_cx = traj['X'][ind]
            traj_x = traj.cx[ind:]
            traj_y = traj.cy[ind:]
            traj = {}
            traj['X'] = traj_x
            traj['Y'] = traj_y

            # pytic.tic()
            spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath, traj)    
            # print('spline generation time: ', pytic.tocvalue())

            pytic.tic()
            # control_steer, self.e, self.theta_e, self.target_ind = self.control_policy_ral.compute_action(current_state_timed,spline_traj)
            control_steer, self.e, self.theta_e, self.target_ind = self.control_policy_ral.compute_action(current_state_timed)
            print('steering control time: ', pytic.tocvalue())
            # reference_control, self.e, self.theta_e, self.target_ind = self.control_policy_ral.compute_action(current_state_timed)
            
            obs_pos = np.array([[self.obs1[0], self.obs1[1]], [self.obs2[0], self.obs2[1]]])
            obs_pos_old = self.obs_pos_past
            obs_pos_new = obs_pos
            obs_vel = (obs_pos_new - obs_pos_old) / self.dt

            obs_center = obs_pos
            obs_r = 0.25
            theta = np.linspace(0, 2*np.pi, 10)
            circ = np.concatenate((np.cos(theta)[:, None], np.sin(theta)[:, None]), axis=1)
            
            obs1 = np.repeat(obs_center[0, :][:, None], 10, axis=1).T + circ * obs_r
            obs2 = np.repeat(obs_center[1, :][:, None], 10, axis=1).T + circ * obs_r
            obs = np.concatenate((obs1, obs2), axis=0)
            agent_state = np.array([self.state.x, self.state.y]).reshape(1, 2)

            obs_dist = np.linalg.norm([agent_state - obs], axis=-1)

            min_obs_dist = np.min(obs_dist)
            
            if min_obs_dist < 1.5:
                
                pytic.tic()
                control_gcbf, next_state, flag = self.control_policy.compute_action(current_state, control_steer, obs=obs, mov_obs_vel=obs_vel)
                # flag = 1
                control_diff = np.linalg.norm([control_steer.acceleration-control_gcbf.acceleration, control_steer.steering_angle-control_gcbf.steering_angle])
                if control_diff < 0.01:
                    flag = 1
                else:
                    flag = 0
            else:
                flag = 1
            
            # print('flag: ', flag)
            # if flag == 0:
            #     print('states: ', next_state[:, :2])
            # flag = 1
            # print('cmarl policy time: ', pytic.tocvalue())
            if flag == 0:
                # continue
                # print('original next state:', next_state[0], next_state[1])
                # next_state_dir = np.array([next_state[0] - self.state.x, next_state[1] - self.state.y])
                # next_state_dir = next_state_dir / np.linalg.norm(next_state_dir)
                
                # goal_dir = np.array([self.goal[0] - self.state.x, self.goal[1] - self.state.y])
                # goal_dir = goal_dir / np.linalg.norm(goal_dir)
                
                next_state = np.array(next_state[:, -20:])
                # print('next step: ', next_state[0], next_state[1])
                # if np.dot(next_state_dir, goal_dir) < 0.9:
                #     next_state[0] = self.state.x + 0.1 * next_state_dir[0]
                #     next_state[1] = self.state.y + 0.1 * next_state_dir[1]
                
                # print('modified next state:', next_state[0], next_state[1])
            
                traj = spline_traj
                # ind, _ = spline_traj.calc_nearest_index(self.state)
                # closest_cx = traj['X'][ind]
                traj_x = traj.cx[-5:]
                traj_y = traj.cy[-5:]

                # c = np.linspace(0, 1, 10)
                
                # x_ref1 = np.hstack([np.array([self.state.x]).squeeze(), next_state[:, 0].squeeze()])
                # y_ref1 = np.hstack([np.array([self.state.y]).squeeze(), next_state[:, 1].squeeze()])
                x_ref1 = next_state[:, 0]
                y_ref1 = next_state[:, 1]
                # x_ref1 = self.state.x * (1 - c) + next_state[0] * c
                # y_ref1 = self.state.y * (1 - c) + next_state[1] * c
                
                x_ref = np.concatenate((x_ref1, np.array(traj_x)))

                y_ref = np.concatenate((y_ref1, np.array(traj_y)))
                # print('x ref: ', x_ref)
                # print('y ref: ', y_ref)
                
                traj = {}
                traj['X'] = x_ref 
                traj['Y'] = y_ref 
                
                pytic.tic()
                spline_traj  = SplineTrajectory2D(self.v_ref,self.traj_filepath, traj)
                
                cx = spline_traj.cx
                cy = spline_traj.cy
                print('current state:', self.state.x, self.state.y)
                print('first c:', cx[0], cy[0])
               
                control_steer, self.e, self.theta_e, self.target_ind = self.control_policy_ral.compute_action(current_state_timed,spline_traj)

                print('second steering time: ', pytic.tocvalue())

                # self.control = control_gcbf
            # else:
            self.control = control_steer
            # else:
            #     self.control = control_gcbf

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
        # msg.drive.mode = 1
        msg.drive.steering_angle = self.control.steering_angle

        msg.drive.acceleration = self.control.acceleration
        msg.drive.acceleration = np.clip(msg.drive.acceleration, -self.accel_limit, self.accel_limit)

        # Control speed rather than acceleration directly
        self.desired_speed += self.dt * (self.control.acceleration)
        # self.desired_speed = self.control.acceleration
        # self.desired_speed = self.state.speed + self.control.acceleration * self.dt

        if self.desired_speed > self.v_ref:
            self.desired_speed = self.v_ref
            msg.drive.acceleration = 0.0
        elif self.desired_speed < -self.v_ref / 2:
            self.desired_speed = -self.v_ref / 2
            msg.drive.acceleration = 0.0

        msg.drive.mode = 0
        msg.drive.speed = self.desired_speed

        # msg.drive.acceleration = np.clip(msg.drive.acceleration, a_min=-0.5, a_max=0.5)

        self.control_pub.publish(msg)
        # print('control:', self.control.steering_angle, self.control.acceleration)
        # print('speed:', self.desired_speed)
        dist_goal = np.sqrt((self.state.x - self.goal[0])**2 + (self.state.y - self.goal[1])**2)
        # print('distance to goal:', dist_goal)

        traj_msg = MultiArray()

        traj_msg.datax = spline_traj.cx
        traj_msg.datay = spline_traj.cy

        self.traj_pub.publish(traj_msg)

        # v = self.state.speed
        # theta = self.state.theta
        # delta = self.control.steering_angle
        # a = self.control.acceleration
        # dq_dt = np.array(
        #             [
        #                 v * np.cos(theta),
        #                 v * np.sin(theta),
        #                 # (v / self.axle_length) * np.tan(delta),
        #                 delta, 
        #                 a,
        #             ]
        #         )
        # if not isinstance(self.state, np.ndarray):
        #     state_np = np.array([self.state.x, self.state.y, self.state.theta, self.state.speed])
        # else:
        #     state_np = self.state
        # self.control_state = state_np + self.dt * dq_dt
        # print('control state: ', self.control_state)
        # print('ekf state: ', state_np)
        print('control: ', msg.drive.speed, msg.drive.acceleration, msg.drive.steering_angle)
        # if self.actual_state is not None:
        #     print('actual state: ', self.actual_state)
        # self.control_state = None
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