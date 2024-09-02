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

import matplotlib.pyplot as plt


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
            # "/vicon/realm_turtle_1/realm_turtle_1",
            # "/vicon/realm_turtle_2/realm_turtle_2",
            "/vicon/realm_obs/realm_obs",
            "/vicon/realm_obs2/realm_obs2"
        ]

        self.traj_topic = rospy.get_param(
            "~traj_topic", "/vesc/high_level/ackermann_cmd_mux/traj"
        )

        self.new_trajx = None
        self.new_trajy = None

        self.traj_sub = rospy.Subscriber(
            self.traj_topic, MultiArray, self.traj_callback
        )

        default_position_names = [
            "f1tenth",
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

    def position_callback(self, msg, idx):
        self.xy[idx,0] = msg.transform.translation.x
        self.xy[idx,1] = msg.transform.translation.y
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
        
        
        pts = ax.scatter(self.xy[:, 0], self.xy[:, 1], animated=True, s=100, c=['b', 'r', 'r'])

        yaw = self.theta[0]
        r = 0.2 
        pt_arrow = ax.arrow(self.xy[0, 0], self.xy[0, 1], r*np.cos(yaw), r*np.sin(yaw), head_width=0.1, head_length=0.1, fc='k', ec='k', animated=True)
        
        obs_pos = self.xy[-2:, :]
        obs_center = obs_pos
        obs_r = 0.5
        theta = np.linspace(0, 2*np.pi, 20)
        circ = np.concatenate((np.cos(theta)[:, None], np.sin(theta)[:, None]), axis=1)
        
        obs1 = np.repeat(obs_center[0, :][:, None], 20, axis=1).T + circ * obs_r
        obs2 = np.repeat(obs_center[1, :][:, None], 20, axis=1).T + circ * obs_r
        obs = np.concatenate((obs1, obs2), axis=0)

        pts_obs = ax.scatter(obs[:, 0], obs[:, 1], animated=True, s=100, c=['r']*40)
        
        pts_obs_center1 = ax.scatter(obs_center[0, 0], obs_center[0, 1], animated=True, s=100, c=['k'])
        pts_obs_center2 = ax.scatter(obs_center[1, 0], obs_center[1, 1], animated=True, s=100, c=['k'])

        (pts1, )= ax.plot(np.array(self.ref_traj.cx), np.array(self.ref_traj.cy), c='k', linestyle='-', animated=True, linewidth=2)

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
        ax.draw_artist(pts_obs_center1)
        ax.draw_artist(pts_obs_center2)
        fig.canvas.blit(fig.bbox)

        while not rospy.is_shutdown():
            fig.canvas.restore_region(bg)
            pts.set_offsets(self.xy)

            obs_pos = self.xy[-2:, :]
            obs_center = obs_pos
            circ = np.concatenate((np.cos(theta)[:, None], np.sin(theta)[:, None]), axis=1)
            
            obs1 = np.repeat(obs_center[0, :][:, None], 20, axis=1).T + circ * obs_r
            obs2 = np.repeat(obs_center[1, :][:, None], 20, axis=1).T + circ * obs_r
            obs = np.concatenate((obs1, obs2), axis=0)
            pts_obs.set_offsets(obs)
            pts_obs_center1.set_offsets(obs_center[0, :])
            pts_obs_center2.set_offsets(obs_center[1, :])
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
            ax.draw_artist(pts_obs_center1)
            ax.draw_artist(pts_obs_center2)

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