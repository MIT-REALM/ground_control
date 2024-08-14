#!/usr/bin/env python3
"""Simulate f1tenth as a ROS node."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Twist
import matplotlib.pyplot as plt
import matplotlib

import os
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D



class VisualizeSimulator:
    """simple visualizer"""

    def __init__(self):
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("simple_visualizer")

        default_position_topics = [
            "/vicon/realm_f1tenth/realm_f1tenth",
            "/vicon/realm_turtle_1/realm_turtle_1",
            "/vicon/realm_turtle_2/realm_turtle_2"
        ]

        default_position_labels = [
            "f1tenth",
            "turtle1",
            "turtle2"
        ]

        self.position_topics = rospy.get_param(
            "~position_topics", default_position_topics
        )

        self.position_labels = rospy.get_param(
            "~position_labels", default_position_labels
        )

        self.xy = np.zeros((len(self.position_topics), 2))

        self.position_subs = [rospy.Subscriber(
            topic, TransformStamped, lambda msg, i=idx: self.position_callback(msg, i)
        ) for idx, topic in enumerate(self.position_topics)]

        self.draw_traj = rospy.get_param("~draw_traj", False)

        self.traj_filepath = os.path.join(
            rospy.get_param("~base_path"), 
            rospy.get_param("~filename")
        )

        if self.draw_traj:
            self.ref_traj = SplineTrajectory2D(0.5,self.traj_filepath)
            self.x_min = min(self.ref_traj.cx)
            self.x_max = max(self.ref_traj.cx)
            self.y_min = min(self.ref_traj.cy)
            self.y_max = max(self.ref_traj.cy)
        else:
            self.ref_traj = None
            self.x_min = rospy.get_param("x_min", -5)
            self.x_max = rospy.get_param("x_max",  5)
            self.y_min = rospy.get_param("y_min", -5)
            self.y_max = rospy.get_param("y_max",  5)

        self.grace = rospy.get_param("grace",  2)


    def position_callback(self, msg, idx):
        self.xy[idx,0] = msg.transform.translation.x
        self.xy[idx,1] = msg.transform.translation.y
        # print(idx, self.xy[idx,:])

    def run(self):
        # see https://matplotlib.org/stable/users/explain/animations/blitting.html
        fig, ax = plt.subplots(figsize=(10, 10))
        pts = ax.scatter(self.xy[:,0], self.xy[:,1], animated=True)

        ax.set_xlim(self.x_min-self.grace, self.x_max+self.grace)
        ax.set_ylim(self.y_min-self.grace, self.y_max+self.grace)

        annos = [ax.annotate(name, xy=self.xy[idx,:], animated=True) 
                 for idx, name in enumerate(self.position_labels)]
        
        if self.draw_traj:
            plt.plot(self.ref_traj.cx, self.ref_traj.cy)
            plt.scatter(self.ref_traj.traj['X'], self.ref_traj.traj['Y'])

        plt.show(block=False)
        plt.pause(0.1)

        bg = fig.canvas.copy_from_bbox(fig.bbox)
        ax.draw_artist(pts)
        fig.canvas.blit(fig.bbox)

        while not rospy.is_shutdown():
            fig.canvas.restore_region(bg)
            pts.set_offsets(self.xy)
            for idx, anno in enumerate(annos):
                # print(idx, anno, self.xy[idx,:])
                anno.set_position(self.xy[idx,:])
                # anno.xy = self.xy[idx,:]
                ax.draw_artist(anno)
            ax.draw_artist(pts)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            # plt.pause(0.01)


if __name__ == "__main__":
    try:
        sim_node = VisualizeSimulator()
        sim_node.run()
    except rospy.ROSInterruptException:
        pass