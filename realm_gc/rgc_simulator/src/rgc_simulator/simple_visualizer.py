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
        rospy.init_node("visualize_simulator")

        default_position_topics = [
            "/vicon/realm_f1tenth/realm_f1tenth",
            "/vicon/realm_turtle_1/realm_turtle_1",
            "/vicon/realm_turtle_2/realm_turtle_2"
        ]

        default_position_names = [
            "f1tenth",
            "turtle1",
            "turtle2"
        ]

        self.position_topics = rospy.get_param(
            "~visualizer_position_topics", default_position_topics
        )

        self.position_names = rospy.get_param(
            "~visualizer_position_names", default_position_names
        )

        self.xy = np.zeros((len(self.position_topics), 2))

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



    def position_callback(self, msg, idx):
        self.xy[idx,0] = msg.transform.translation.x
        self.xy[idx,1] = msg.transform.translation.y
        # print(idx, self.xy[idx,:])

    def run(self):
        # see https://matplotlib.org/stable/users/explain/animations/blitting.html
        fig, ax = plt.subplots(figsize=(10, 10))
        pts = ax.scatter(self.xy[:,0], self.xy[:,1], animated=True)

        x_min = min(self.ref_traj.cx)
        x_max = max(self.ref_traj.cx)
        y_min = min(self.ref_traj.cy)
        y_max = max(self.ref_traj.cy)
        grace = 2

        ax.set_xlim(x_min-grace, x_max+grace)
        ax.set_ylim(y_min-grace, y_max+grace)

        annos = [ax.annotate(name, xy=self.xy[idx,:], animated=True) 
                 for idx, name in enumerate(self.position_names)]
        
        ref_x = [0.0, 0.5, 1.0]
        ref_y = [0.0, 0.5, 1.0]
        
        #plt.plot(self.ref_traj.cx, self.ref_traj.cy)
        #plt.scatter(self.ref_traj.traj['X'], self.ref_traj.traj['Y'])
        plt.scatter(ref_x,ref_y)

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