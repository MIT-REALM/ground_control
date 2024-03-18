"""Define a base class for control."""
import abc
import os

import rospy
from std_msgs.msg import Empty


class RobotControl(abc.ABC):
    def __init__(self):
        rospy.init_node("robot_control_node", anonymous=True)

        # Sets control loop frequency to 30 Hz
        self.rate = rospy.get_param("~rate", 30)  # Hz
        self.dt = 1.0 / self.rate
        self.rate_timer = rospy.Rate(self.rate)

        # Set duration for experiment
        self.T = rospy.get_param("~T", 6)  # seconds
        self.time_begin = rospy.Time.now()

        # Control publisher - to be defined in subclasses, as needed
        self.control_pub = None

        # Initialize the control as stopped
        self.ctrl_c = True

        # Create a subscriber for starting/stopping the controller
        self.start_sub = rospy.Subscriber(
            "/start_control", Empty, self.start_control_callback
        )
        self.stop_sub = rospy.Subscriber(
            "/stop_control", Empty, self.stop_control_callback
        )

        # Load trajectory from file using rosparam.
        self.eqx_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"),
            rospy.get_param("~trajectory/filename"),
        )

        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        self.ctrl_c = True

    def start_control_callback(self, msg):
        """Start the control."""
        self.ctrl_c = False
        self.time_begin = rospy.Time.now()

    def stop_control_callback(self, msg):
        """Stop the control."""
        self.ctrl_c = True

    @abc.abstractmethod
    def reset_control(self, msg=None):
        """Reset the control to stop the experiment and publish the command."""
        return

    @abc.abstractmethod
    def update(self):
        """Update the control policy using latest state estimate and publish it."""
        return

    def run(self):
        """
        Run the control, updating and publishing the control at the specified rate.
        """
        self.reset_control(None)
        while not rospy.is_shutdown():
            # If experiment is not stopped, update the control; otherwise, reset it
            if not self.ctrl_c:
                self.update()
            else:
                self.reset_control(None)

            self.rate_timer.sleep()


if __name__ == "__main__":
    rospy.logerror("This is a base class and should not be run directly.")
