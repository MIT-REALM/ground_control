"""Define a base class for control."""
import abc
import os

import rospy


class RobotControl(abc.ABC):
    def __init__(self):
        rospy.init_node("robot_control_node", anonymous=True)

        # Sets control loop frequency to 30 Hz
        self.rate = rospy.get_param("~rate", 30)  # Hz
        self.dt = 1.0 / self.rate
        self.rate_timer = rospy.Rate(self.rate)

        # Control publisher - to be defined in subclasses, as needed
        self.control_pub = None

        # Initialize the control
        self.reset_control()
        self.ctrl_c = False

        # Load trajectory from file using rosparam.
        self.eqx_filepath = os.path.join(
            rospy.get_param("~trajectory/base_path"),
            rospy.get_param("~trajectory/filename"),
        )

        rospy.on_shutdown(self.shutdownhook)

    def shutdownhook(self):
        self.ctrl_c = True

    @abc.abstractmethod
    def reset_control(self, msg=None):
        """Reset the control."""
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
            self.update()  # Update the control information and publish
            self.rate_timer.sleep()


if __name__ == "__main__":
    rospy.loginfo("This is a base class and should not be run directly.")