#!/usr/bin/env python3
"""Simulate f1tenth as a ROS node."""
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped, Twist
import matplotlib.pyplot as plt
import matplotlib


class VisualizeSimulator:
    """simple visualizer"""

    def __init__(self):
        """Initialize the simulator."""
        # Initialize the node
        rospy.init_node("visualize_simulator")

        self.position_topics = [
            "/vicon/realm_f1tenth/realm_f1tenth",
            "/vicon/realm_turtle_1/realm_turtle_1",
            "/vicon/realm_turtle_2/realm_turtle_2"
        ]

        self.position_names = [
            "f1tenth",
            "turtle1",
            "turtle2"
        ]

        self.xy = np.zeros((len(self.position_topics), 2))

        self.position_subs = [rospy.Subscriber(
            topic, TransformStamped, lambda msg, i=idx: self.position_callback(msg, i)
        ) for idx, topic in enumerate(self.position_topics)]

    def position_callback(self, msg, idx):
        self.xy[idx,0] = msg.transform.translation.x
        self.xy[idx,1] = msg.transform.translation.y
        print(idx, self.xy[idx,:])

    def run(self):
        fig, ax = plt.subplots()
        pts = ax.scatter(self.xy[:,0], self.xy[:,1], animated=True)
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)

        annos = [ax.annotate(name, xy=self.xy[idx,:], animated=True) 
                 for idx, name in enumerate(self.position_names)]
        
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
