"""Define a policy for trajectory tracking."""
from dataclasses import dataclass

from rgc_control.policies.policy import ControlAction, ControlPolicy
from rgc_control.policies.tracking.steering_policies import (
    Pose2DObservation,
    SteeringObservation,
)
from rgc_control.policies.tracking.trajectory import LinearTrajectory2D


@dataclass
class TimedPose2DObservation(Pose2DObservation):
    """The observation for a single robot's 2D pose with a timestamp."""

    t: float


class TrajectoryTrackingPolicy(ControlPolicy):
    """Tracks a trajectory given a steering controller.

    args:
        trajectory: the trajectory to track
        controller: the controller to use to steer towards waypoints
    """

    trajectory: LinearTrajectory2D
    steering_controller: ControlPolicy

    def __init__(
        self, trajectory: LinearTrajectory2D, steering_controller: ControlPolicy
    ):
        self.trajectory = trajectory
        self.steering_controller = steering_controller

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
        steering_observation = SteeringObservation(
            pose=observation,
            goal=Pose2DObservation(
                x=waypoint[0],
                y=waypoint[1],
                theta=0.0,
                v=0.0,
            ),
        )
        return self.steering_controller.compute_action(steering_observation)
