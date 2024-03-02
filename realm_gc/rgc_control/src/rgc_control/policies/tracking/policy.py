"""Define a policy for trajectory tracking."""
from rgc_control.policies.policy import ControlAction, ControlPolicy, Observation


class TimedPose2DObservation(Pose2DObservation):
    """The observation for a single robot's 2D pose with a timestamp."""

    t: float


class TrackingObservation(Observation):
    """The observation for a single robot's 2D pose relative to some goal."""

    pose: Pose2DObservation
    goal: Pose2DObservation


class TrajectoryTrackingPolicy(ControlPolicy):
    """Tracks a trajectory given a steering controller.

    args:
        trajectory: the trajectory to track
        controller: the controller to use to steer towards waypoints
    """

    trajectory: MultiAgentTrajectoryLinear
    steering_controller: ControlPolicy

    @property
    def observation_type(self):
        return TimedPose2DObservation

    @property
    def action_type(self):
        return self.steering_controller.action_type

    def compute_action(self, observation: TimedPose2DObservation) -> ControlAction:
        """Takes in an observation and returns a control action."""
        # Compute the desired waypoint
        waypoint = self.trajectory(observation.t)

        # Compute the control action to steer towards the waypoint
        return self.steering_controller.compute_action(observation, waypoint)
        return self.steering_controller.compute_action(observation, waypoint)
