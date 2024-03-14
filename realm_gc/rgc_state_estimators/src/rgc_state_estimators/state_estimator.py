"""Define a base class for state estimators."""
import abc

import rospy
from std_msgs.msg import Empty


class StateEstimator(abc.ABC):
    """
    Base class for state estimation nodes.
    """

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("state_estimation_node")

        # Fetch parameters
        self.rate = rospy.get_param("~rate", 10)  # Hz
        self.dt = 1.0 / self.rate
        self.rate_timer = rospy.Rate(self.rate)

        # Publisher - to be defined in subclasses, as needed
        # e.g., self.pub = rospy.Publisher(
        #           f"{rospy.get_name()}/estimate",
        #           PoseStamped,
        #           queue_size=10
        #       )
        self.pub = None

        # Reset event subscriber
        self.reset_sub = rospy.Subscriber(
            f"{rospy.get_name()}/reset", Empty, self.reset_state
        )

        # Other subscribers - to be defined in subclasses, as needed
        # e.g., rospy.Subscriber("input_topic", MessageType, self.callback_method)

        # Initialize the state
        self.reset_state()

    @abc.abstractmethod
    def reset_state(self, msg=None):
        """Reset the state of the estimator."""
        return

    @abc.abstractmethod
    def update(self):
        """Update the filter state and publish the new state estimate."""
        return

    def run(self):
        """
        Run the estimator, updating and publishing the state at the specified rate.
        """
        self.reset_state(None)
        while not rospy.is_shutdown():
            self.update()  # Update the state estimate and publish
            self.rate_timer.sleep()


if __name__ == "__main__":
    # This base class should not be run directly since it includes abstract methods.
    # Instead, it should be subclassed, and those methods should be implemented.
    rospy.loginfo("This is a base class and should not be run directly.")
