#!/usr/bin/env python3
"""Simulate f1tenth as a ROS node."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Twist
# import tkinter
# import matplotlib
from f1tenth_msgs.msg import MultiArray
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

from rgc_state_estimators.msg import F1TenthState

import matplotlib.pyplot as plt
import jax.numpy as jnp

import os
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D


class VisualizeSimulator:
    """simple visualizer"""

    def __init__(self):
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("visualize_simulator")

        default_position_topics = [
            "/vicon/realm_f1tenth/realm_f1tenth",
            "/vicon/realm_obs/realm_obs",
            "/vicon/realm_obs2/realm_obs2",
            "/vicon/realm_turtle_1/realm_turtle_1",
            "/vicon/realm_turtle_2/realm_turtle_2",
            
        ]

        self.traj_topic = rospy.get_param(
            "~traj_topic", "/vesc/high_level/ackermann_cmd_mux/traj"
        )
        # self.dt = 0.005

        self.goal_topic = rospy.get_param(
            "~goal_topic", "/vesc/high_level/ackermann_cmd_mux/goal")

        self.goal_sub = rospy.Subscriber(
            self.goal_topic, F1TenthState, self.goal_callback
        )

        # self.goal_pub = rospy.Publisher(
        #     "/vesc/high_level/ackermann_cmd_mux/goal",
        #     F1TenthState,
        #     queue_size=1,
        # )

        self.new_trajx = None
        self.new_trajy = None
        self.goal = F1TenthState()

        self.goal.x = 0.0
        self.goal.y = -4.0

        self.traj_sub = rospy.Subscriber(
            self.traj_topic, MultiArray, self.traj_callback
        )

        default_position_names = [
            "f1tenth",
            "obs1",
            "obs2",
            "turtle1",
            "turtle2",
        ]

        self.position_topics = rospy.get_param(
            "~visualizer_position_topics", default_position_topics
        )

        self.position_names = rospy.get_param(
            "~visualizer_position_names", default_position_names
        )

        self.xy = np.zeros((len(self.position_topics), 2))

        self.theta = np.zeros(len(self.position_topics))

        self.position_subs = [rospy.Subscriber(
            topic, TransformStamped, lambda msg, i=idx: self.position_callback(msg, i)
        ) for idx, topic in enumerate(self.position_topics)]

        self.traj_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"), 
            rospy.get_param("~trajectory/filename")
        )

        self.ref_traj = SplineTrajectory2D(0.5,self.traj_filepath)
        # print(self.ref_traj.cx)
        # print(self.ref_traj.cy)

    def traj_callback(self, msg):
        
        self.new_trajx = msg.datax
        self.new_trajy = msg.datay
    
    def goal_callback(self, msg):
        self.goal = msg
        self.goal.x = self.goal.x - 3.5
        self.goal.y = self.goal.y - 5.0
        # self.new_trajy = msg.datay

    def position_callback(self, msg, idx):
        self.xy[idx,0] = msg.transform.translation.x #- 3.5
        self.xy[idx,1] = msg.transform.translation.y #- 5.0
        z = msg.transform.rotation.z
        w = msg.transform.rotation.w
        self.theta[idx] = np.arctan2(2*(w*z), 1-2*z**2)
        # print(idx, self.xy[idx,:])

    def run(self):

        # metadata = dict(title='Movie Test', artist='Matplotlib',
        #         comment='Movie support!')
        # writer = FFMpegWriter(fps=15, metadata=metadata)
        # see https://matplotlib.org/stable/users/explain/animations/blitting.html
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # goals = jnp.array([self.goal.x, self.goal.y, self.goal.theta, self.goal.speed]).reshape(1, 4)
        #     # goals = goals.at[1].set(goals[1] + 5.5)
        #     # goals = goals.at[0].set(goals[0] + 3.5)

        # thetas = jnp.arctan2(goals[:, 1] - 7.0 / 2, goals[:, 0] - 7.0 / 2)
        # thetas_next = thetas + 1.0 * self.dt / 2.5
        # next_goal_pos = jnp.stack([7.0 / 2 + 2.5 * jnp.cos(thetas_next),
        #                         7.0 / 2 + 2.5 * jnp.sin(thetas_next)], axis=-1)
        # next_goal_vel_dir = jnp.stack([-jnp.sin(thetas_next), jnp.cos(thetas_next)], axis=-1)
        # next_goal_vel = jnp.ones((1,)) * 1.0
        # next_goals = goals.at[:, :2].set(next_goal_pos).at[:, 2:4].set(next_goal_vel_dir).at[:, 4].set(next_goal_vel)
        
        
        # next_goals = next_goals.squeeze()
        # goal_msg = F1TenthState()
        # goal_msg.x = next_goals[0]
        # goal_msg.y = next_goals[1]
        # goal_msg.theta = next_goals[2]
        # goal_msg.speed = next_goals[3]

        # self.goal = goal_msg

        # self.goal_pub.publish(goal_msg)
            
        pts = ax.scatter(self.xy[:, 0], self.xy[:, 1], animated=True, s=100, c=['b', 'r', 'r', 'r', 'r'])

        yaw = self.theta[0]
        r = 0.2 
        pt_arrow = ax.arrow(self.xy[0, 0], self.xy[0, 1], r*np.cos(yaw), r*np.sin(yaw), head_width=0.1, head_length=0.1, fc='k', ec='k', animated=True)
        
        obs_pos = self.xy[1:, :]
        obs_center = obs_pos
        obs_r = 0.0
        theta = np.linspace(0, 2*np.pi, 10)
        circ = np.concatenate((np.cos(theta)[:, None], np.sin(theta)[:, None]), axis=1)
        
        obs1 = np.repeat(obs_center[0, :][:, None], 10, axis=1).T + circ * obs_r
        obs2 = np.repeat(obs_center[1, :][:, None], 10, axis=1).T + circ * obs_r
        obs3 = np.repeat(obs_center[2, :][:, None], 10, axis=1).T + circ * obs_r
        obs4 = np.repeat(obs_center[3, :][:, None], 10, axis=1).T + circ * obs_r
        
        obs = np.concatenate((obs1, obs2, obs3, obs4), axis=0)

        pts_obs = ax.scatter(obs[:, 0], obs[:, 1], animated=True, s=100, c=['r']*40)

        (pts1, )= ax.plot(np.array(self.ref_traj.cx), np.array(self.ref_traj.cy), c='k', linestyle='-', animated=True, linewidth=2)

        pts_goal = ax.scatter(self.goal.x, self.goal.y, animated=True, s=100, c='k')
        # lines = ax.plot(self.xy, self.xy + 0.1*np.array([np.cos(self.theta), np.sin(self.theta)]).T, animated=True, linewidth=2)
        
        x_min = -5
        x_max = 5
        y_min = -5
        y_max = 5
        # x_min = min(self.ref_traj.cx)
        # x_max = max(self.ref_traj.cx)
        # y_min = min(self.ref_traj.cy)
        # y_max = max(self.ref_traj.cy)
        grace = 2

        ax.set_xlim(x_min-grace, x_max+grace)
        ax.set_ylim(y_min-grace, y_max+grace)

        annos = [ax.annotate(name, xy=self.xy[idx,:], animated=True) 
                 for idx, name in enumerate(self.position_names)]
        
        # with writer.saving(fig, "writer_test.mp4", 100):    
        # plt.plot(self.ref_traj.cx, self.ref_traj.cy)
        # plt.scatter(self.ref_traj.traj['X'], self.ref_traj.traj['Y'])
        # plt.scatter(ref_x,ref_y)

        plt.show(block=False)
        plt.pause(0.1)

        bg = fig.canvas.copy_from_bbox(fig.bbox)
        ax.draw_artist(pts)
        ax.draw_artist(pts1)
        ax.draw_artist(pts_obs)
        ax.draw_artist(pt_arrow)
        ax.draw_artist(pts_goal)
        fig.canvas.blit(fig.bbox)

        while not rospy.is_shutdown():
            # print("printing xy shape: ", self.xy.shape)
            
            # goals = jnp.array([self.goal.x, self.goal.y, self.goal.theta, self.goal.speed]).reshape(1, 4)
            # # goals = goals.at[1].set(goals[1] + 5.5)
            # # goals = goals.at[0].set(goals[0] + 3.5)

            # thetas = jnp.arctan2(goals[:, 1] - 7.0 / 2, goals[:, 0] - 7.0 / 2)
            # thetas_next = thetas + 1.0 * self.dt / 2.5
            # next_goal_pos = jnp.stack([7.0 / 2 + 2.5 * jnp.cos(thetas_next),
            #                         7.0 / 2 + 2.5 * jnp.sin(thetas_next)], axis=-1)
            # next_goal_vel_dir = jnp.stack([-jnp.sin(thetas_next), jnp.cos(thetas_next)], axis=-1)
            # next_goal_vel = jnp.ones((1,)) * 1.0
            # next_goals = goals.at[:, :2].set(next_goal_pos).at[:, 2:4].set(next_goal_vel_dir).at[:, 4].set(next_goal_vel)
            
            
            # next_goals = next_goals.squeeze()
            # goal_msg = F1TenthState()
            # goal_msg.x = next_goals[0]
            # goal_msg.y = next_goals[1]
            # goal_msg.theta = next_goals[2]
            # goal_msg.speed = next_goals[3]

            # self.goal = goal_msg

            # self.goal_pub.publish(goal_msg)

            fig.canvas.restore_region(bg)
            pts.set_offsets(self.xy)
            pts_goal.set_offsets(np.array([self.goal.x, self.goal.y]))

            obs_pos = self.xy[1:, :]
            obs_center = obs_pos
            obs_r = 0.0
            theta = np.linspace(0, 2*np.pi, 10)
            circ = np.concatenate((np.cos(theta)[:, None], np.sin(theta)[:, None]), axis=1)
            
            obs1 = np.repeat(obs_center[0, :][:, None], 10, axis=1).T + circ * obs_r
            obs2 = np.repeat(obs_center[1, :][:, None], 10, axis=1).T + circ * obs_r
            obs3 = np.repeat(obs_center[2, :][:, None], 10, axis=1).T + circ * obs_r
            obs4 = np.repeat(obs_center[3, :][:, None], 10, axis=1).T + circ * obs_r
        
            obs = np.concatenate((obs1, obs2, obs3, obs4), axis=0)
            pts_obs.set_offsets(obs)

            # pts1.set_offsets(np.array([self.new_trajx, self.new_trajy]))
            # pts1 = ax.scatter(self.new_trajx, self.new_trajy, animated=True, c='k', linestyle='-')
            for idx, anno in enumerate(annos):
                # print(idx, anno, self.xy[idx,:])
                anno.set_position(self.xy[idx,:])
                # anno.xy = self.xy[idx,:]
                ax.draw_artist(anno)
                # draw orientation as straight line
                # ax.plot([self.xy[idx,0], self.xy[idx,0] + 0.1*np.cos(self.theta[idx])], 
                #         [self.xy[idx,1], self.xy[idx,1] + 0.1*np.sin(self.theta[idx])])
            # if self.new_trajx is not None and self.new_trajy is not None:
            #     ax.plot(self.new_trajx, self.new_trajy, '--r')
            # print('type new trajx: ', type(self.new_trajx))
            if self.new_trajx is not None and self.new_trajy is not None:
                pts1.set_xdata(self.new_trajx)
                pts1.set_ydata(self.new_trajy)
                ax.draw_artist(pts1)
            
            # pt_arrow.set_data([self.xy[0, 0], self.xy[0, 1], r*np.cos(self.theta[0]), r*np.sin(self.theta[0])])
            pt_arrow.set_data(x = self.xy[0, 0], y=self.xy[0, 1], dx=r*np.cos(self.theta[0]), dy=r*np.sin(self.theta[0]))
            ax.draw_artist(pts)
            ax.draw_artist(pts_obs)
            ax.draw_artist(pt_arrow)
            ax.draw_artist(pts_goal)
            # ax.draw_artist(lines)
            # writer.grab_frame()
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            

if __name__ == "__main__":
    try:
        sim_node = VisualizeSimulator()
        sim_node.run()
    except rospy.ROSInterruptException:
        pass