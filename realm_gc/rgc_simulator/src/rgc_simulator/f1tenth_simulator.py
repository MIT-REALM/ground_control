#!/usr/bin/env python3
"""Simulate f1tenth as a ROS node."""
import numpy as np
import rospy
from f1tenth_msgs.msg import F1TenthDriveStamped
from geometry_msgs.msg import TransformStamped
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D
from rgc_state_estimators.msg import F1TenthState
import os

class F1TenthSimulator:
    """Implement a simple simulator for the f1tenth with vicon (no other sensors)."""

    def __init__(self):
        sim_rate = os.environ['SIMRATE']
        print('sim_rate: ', sim_rate)
        if sim_rate is None:
            sim_rate = 10
        else:
            sim_rate = int(sim_rate)
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("f1tenth_simulator")
        self.axle_length = rospy.get_param("~axle_length", 0.28)
        self.control_topic = rospy.get_param(
            "~control_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
        )
        self.sim_position_topic = rospy.get_param(
            "~position_topic_sim", "/vicon/realm_f1tenth/realm_f1tenth_sim"
        )

        self.position_topic = rospy.get_param(
            "~position_topic", "/vicon/realm_f1tenth/realm_f1tenth"
        )

        self.position_obs1= rospy.get_param(
            "~position_topic1", "/vicon/realm_obs/realm_obs"
        )

        self.position_obs2= rospy.get_param("~position_topic2", "/vicon/realm_obs2/realm_obs2")

        # Initialize the f1tenth state
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.obs = np.array([0.0, -0.5, 0.0, 0.0])
        self.obs2 = np.array([0.0, 1.0, 0.0, 0.0])
        self.command = np.array([0.0, 0.0, 0.0])

        # Set the simulation rate
        self.rate_hz = rospy.get_param("~rate", sim_rate)
        self.rate = rospy.Rate(self.rate_hz)

        self.obs_1_mv = 1
        self.obs_2_mv = -1
        # Subscribe to cmd_vel
        self.cmd_vel_sub = rospy.Subscriber(
            self.control_topic, F1TenthDriveStamped, self.cmd_callback
        )

        
        self.sim_f1tenth_pub = rospy.Publisher(
            self.sim_position_topic, F1TenthState, queue_size=10)
            
        # Publish the transform of the f1tenth
        self.tf_pub = rospy.Publisher(
            self.position_topic, TransformStamped, queue_size=10
        )

        self.tf_pub1 = rospy.Publisher(
            self.position_obs1, TransformStamped, queue_size=10
        )

        self.tf_pub2 = rospy.Publisher(
            self.position_obs2, TransformStamped, queue_size=10
        )

        self.traj_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"), 
            rospy.get_param("~trajectory/filename")
        )

        self.ref_traj = SplineTrajectory2D(0.5,self.traj_filepath)

        self.state[0] = self.ref_traj.cx[0]
        self.state[1] = self.ref_traj.cy[0]
        self.state[2] = self.ref_traj.cyaw[0]

        tf_obs = TransformStamped()
        tf_obs.header.stamp = rospy.Time.now()
        tf_obs.header.frame_id = "world"
        tf_obs.child_frame_id = "realm_obs"
        tf_obs.transform.translation.x = self.obs[0]
        tf_obs.transform.translation.y = self.obs[1]
        tf_obs.transform.translation.z = 0.0
        tf_obs.transform.rotation.x = 0.0
        tf_obs.transform.rotation.y = 0.0
        tf_obs.transform.rotation.z = 0.0
        tf_obs.transform.rotation.w = 0.0

        tf_obs2 = TransformStamped()
        tf_obs2.header.stamp = rospy.Time.now()
        tf_obs2.header.frame_id = "world"
        tf_obs2.child_frame_id = "realm_obs2"

        tf_obs2.transform.translation.x = self.obs2[0]
        tf_obs2.transform.translation.y = self.obs2[1]
        tf_obs2.transform.translation.z = 0.0
        tf_obs2.transform.rotation.x = 0.0
        tf_obs2.transform.rotation.y = 0.0
        tf_obs2.transform.rotation.z = 0.0
        tf_obs2.transform.rotation.w = 0.0
    
        self.tf_pub1.publish(tf_obs)
        self.tf_pub2.publish(tf_obs2)
        self.previous_command = self.command
        

    def cmd_callback(self, msg):
        """Update the saved command."""
        self.command = np.array([msg.drive.steering_angle, msg.drive.acceleration, msg.drive.speed])
        # print('command: ', self.command)

    def run(self):
        
        """Run the simulation."""
        # while self.command is None:
        #         rospy.loginfo(
        #             "Waiting for command"
        #         )
        #         rospy.sleep(1.0)

        # rospy.loginfo("Command received. Starting simulation.")
        while not rospy.is_shutdown():
            # Update the state
            x, y, theta, v = self.state
            delta, a, v_des= self.command
            # if v_des <= 0:
            #     v = 0
            # if v_des > 1.5:
            #     v_des = 1.5
            # print('state before sim: ', self.state)
            # print('command: ', self.command)
            new_command = self.command

            diff_command = np.linalg.norm(self.previous_command - new_command)
            if diff_command < 1e-6:
                dq_dt = self.state * 0.0
            else:
                dq_dt = np.array(
                    [
                        v * np.cos(theta),
                        v * np.sin(theta),
                        # (v / self.axle_length) * np.tan(delta),
                        delta, 
                        a,
                    ]
                )
                self.previous_command = self.command
            # print('delta in sim: ', delta)
            # print('a in sim: ', a)

            self.state += dq_dt / self.rate_hz
            # self.state[3] = max(0, self.state[3])
            self.state[2] = self.state[2] % (2 * np.pi)
            # print('state after sim: ', self.state)
            # print('state: ', self.state)
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

            obs_speed = 0.001
            if self.obs_1_mv == 1:
                self.obs[0] += obs_speed
                if self.obs[0] >= 0.3:
                    self.obs_1_mv = -1
            else:
                self.obs[0] -= obs_speed
                if self.obs[0] <= -0.3:
                    self.obs_1_mv = 1

            tf_obs = TransformStamped()
            tf_obs.header.stamp = rospy.Time.now()
            tf_obs.header.frame_id = "world"
            tf_obs.child_frame_id = "realm_obs"
            tf_obs.transform.translation.x = self.obs[0]
            tf_obs.transform.translation.y = self.obs[1]
            tf_obs.transform.translation.z = 0.0
            tf_obs.transform.rotation.x = 0.0
            tf_obs.transform.rotation.y = 0.0
            tf_obs.transform.rotation.z = 0.0
            tf_obs.transform.rotation.w = 0.0

            tf_obs2 = TransformStamped()
            tf_obs2.header.stamp = rospy.Time.now()
            tf_obs2.header.frame_id = "world"
            tf_obs2.child_frame_id = "realm_obs2"

            if self.obs_2_mv == 1:
                self.obs2[0] += obs_speed
                if self.obs2[0] >= 0.3:
                    self.obs_2_mv = -1
            else:
                self.obs2[0] -= obs_speed
                if self.obs2[0] <= -0.3:
                    self.obs_2_mv = 1

            tf_obs2.transform.translation.x = self.obs2[0]
            tf_obs2.transform.translation.y = self.obs2[1]
            tf_obs2.transform.translation.z = 0.0
            tf_obs2.transform.rotation.x = 0.0
            tf_obs2.transform.rotation.y = 0.0
            tf_obs2.transform.rotation.z = 0.0
            tf_obs2.transform.rotation.w = 0.0

            sim_state = F1TenthState()
            sim_state.x = self.state[0]
            sim_state.y = self.state[1]
            sim_state.theta = self.state[2]
            sim_state.speed = self.state[3]

            self.tf_pub.publish(tf)
            self.tf_pub1.publish(tf_obs)
            self.tf_pub2.publish(tf_obs2)
            self.sim_f1tenth_pub.publish(sim_state)
            # Sleep
            self.rate.sleep()
            self.rate.sleep()
            # rospy.spin()

if __name__ == "__main__":
    try:
        sim_node = F1TenthSimulator()
        sim_node.run()
    except rospy.ROSInterruptException:
        pass
