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


if __name__ == "__main__":
    # Test trajectory tracking control
    import jax.numpy as jnp
    import matplotlib
    import numpy as np

    from rgc_control.policies.tracking.steering_policies import F1TenthSteeringPolicy

    matplotlib.use("Agg")

    # Define a linear trajectory
    trajectory = LinearTrajectory2D(
        p=jnp.array(
            [
                [3 * 0.0, 0.0],
                [3 * 1.0, 1.0],
                [3 * 2.0, 1.0],
                [3 * 3.0, -1.0],
            ]
        )
    )

    # Define a steering controller and tracking controller
    steering_controller = F1TenthSteeringPolicy(
        np.array([0.0, 0.0, 0.0, 0.1]), 0.28, 0.05
    )
    tracking_controller = TrajectoryTrackingPolicy(
        trajectory=trajectory,
        steering_controller=steering_controller,
    )

    # Simulate
    steps = 700
    dt = 0.05
    ts = np.linspace(0, 1, steps)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    states = [state.tolist()]
    for i in range(steps):
        t = ts[i]
        obs = TimedPose2DObservation(
            x=state[0],
            y=state[1],
            theta=state[2],
            v=state[3],
            t=t,
        )
        action = tracking_controller.compute_action(obs)
        state = np.array(
            [
                state[0] + state[3] * np.cos(state[2]) * dt,
                state[1] + state[3] * np.sin(state[2]) * dt,
                state[2] + (state[3] / 0.28) * np.tan(action.steering_angle) * dt,
                state[3] + action.acceleration * dt,
            ]
        )
        states.append(state.tolist())

    # Plot the trajectory
    states = np.array(states)
    import matplotlib.pyplot as plt

    plt.plot(states[:, 0], states[:, 1], "b-", label="Simulated")
    plt.plot(trajectory.p[:, 0], trajectory.p[:, 1], "ro--", label="Desired")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("src/realm_gc/debug.png")
