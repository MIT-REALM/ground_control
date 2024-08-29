"""Define a policy for trajectory tracking."""
from dataclasses import dataclass

import rospy
import jax
import jax.numpy as jnp
import numpy as np
from rgc_control.policies.policy import ControlAction, ControlPolicy
from rgc_control.policies.tracking.steering_policies import (
    Pose2DObservation,
    SteeringObservation,
    G2CGoal2DObservation,
    G2CPose2DObservation
)
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D
from rgc_control.policies.cubic_spline import CubicSpline2D

@dataclass
class TimedPose2DObservation(Pose2DObservation):
    """The observation for a single robot's 2D pose with a timestamp."""

    t: float
@dataclass
class TimedG2CPose2DObservation(G2CPose2DObservation):
    """The observation for a single robot's 2D pose with a timestamp."""

    t: float

# class TrajectoryTrackingPolicy(ControlPolicy):
#     """Tracks a trajectory given a steering controller.

#     args:
#         trajectory: the trajectory to track
#         controller: the controller to use to steer towards waypoints
#     """

#     trajectory: LinearTrajectory2D
#     steering_controller: ControlPolicy

#     def __init__(
#         self, trajectory: LinearTrajectory2D, steering_controller: ControlPolicy
#     ):
#         self.trajectory = trajectory
#         self.steering_controller = steering_controller

#     @property
#     def observation_type(self):
#         return TimedPose2DObservation

#     @property
#     def action_type(self):
#         return self.steering_controller.action_type

#     def compute_action(self, observation: TimedPose2DObservation) -> ControlAction:
#         """Takes in an observation and returns a control action."""
#         # Compute the desired waypoint and tangent vector
#         waypoint = self.trajectory(observation.t)
#         tangent = jax.jit(jax.jacfwd(self.trajectory))(observation.t)

#         # Compute the angle to steer along
#         theta = jnp.pi / 2
#         if jnp.linalg.norm(tangent) >= 0.05:
#             theta = jnp.arctan2(tangent[1], tangent[0])

#         # Compute the control action to steer towards the waypoint
#         steering_observation = SteeringObservation(
#             pose=observation,
#             goal=Pose2DObservation(
#                 x=waypoint[0],
#                 y=waypoint[1],
#                 theta=theta,
#                 v=0.0,
#             ),
#         )
#         return self.steering_controller.compute_action(steering_observation)


class G2CTrajectoryTrackingPolicy(ControlPolicy):
    """Tracks a trajectory given a speed+steering controller.
    args:
        trajectory: the trajectory to track
        controller: the controller to use to steer towards waypoints
    """
    trajectory: SplineTrajectory2D
    steering_controller: ControlPolicy

    def __init__(
        self, trajectory: SplineTrajectory2D, steering_controller: ControlPolicy
    ):
        self.trajectory = trajectory
        self.steering_controller = steering_controller

    @property
    def observation_type(self):
        return TimedPose2DObservation

    @property
    def action_type(self):
        return self.steering_controller.action_type
    
    def compute_action(self, observation: TimedG2CPose2DObservation, traj=None) -> ControlAction:
        """Takes in an observation and returns a control action."""
        # Compute the desired waypoint. 
        #Each waypoint consists of desired x,y,theta & curvature 
        waypoint = self.trajectory(observation.t) 
        
        # Compute the control action to steer towards the waypoint
        steering_observation = SteeringObservation(
            pose=observation,
            goal=G2CGoal2DObservation(
                x=waypoint[0],
                y=waypoint[1],
                theta=waypoint[2],
                v=waypoint[3],
                k=waypoint[4],
            ),
        )
        return self.steering_controller.compute_action(steering_observation, traj)
