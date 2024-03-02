"""Define policies for steering different dynamical systems towards waypoints."""
from dataclasses import dataclass

import numpy as np
import scipy

from rgc_control.policies.policy import ControlAction, ControlPolicy, Observation


@dataclass
class Pose2DObservation(Observation):
    """The observation for a single robot's 2D pose and linear speed."""

    x: float
    y: float
    theta: float


@dataclass
class SteeringObservation(Observation):
    """The observation for a single robot's 2D pose relative to some goal."""

    pose: Pose2DObservation
    goal: Pose2DObservation


@dataclass
class TurtlebotSteeringAction(ControlAction):
    """The action for a turtlebot steering controller."""

    linear_velocity: float
    angular_velocity: float


class TurtlebotSteeringPolicy(ControlPolicy):
    """Steer a turtlebot towards a waypoint using a proportional controller."""

    @property
    def observation_type(self):
        return SteeringObservation

    @property
    def action_type(self):
        return TurtlebotSteeringAction

    def compute_action(
        self, observation: SteeringObservation
    ) -> TurtlebotSteeringAction:
        """Takes in an observation and returns a control action."""
        # Compute the error in the turtlebot's frame
        error = np.array(
            [
                observation.goal.x - observation.pose.x,
                observation.goal.y - observation.pose.y,
            ]
        ).reshape(-1, 1)
        error = (
            np.array(
                [
                    [np.cos(observation.pose.theta), -np.sin(observation.pose.theta)],
                    [np.sin(observation.pose.theta), np.cos(observation.pose.theta)],
                ]
            ).T  # Transpose to rotate into turtlebot frame
            @ error
        )

        # Compute the control action
        linear_velocity = 0.5 * error[0]  # projection along the turtlebot x-axis

        # Compute the angular velocity: steer towards the goal if we're far from it
        # (so the arctan is well defined), and align to the goal orientation
        if linear_velocity > 0.05:
            angular_velocity = np.arctan2(error[1], error[0])
        else:
            angle_error = observation.goal.theta - observation.pose.theta
            if angle_error > np.pi:
                angle_error -= 2 * np.pi
            if angle_error < -np.pi:
                angle_error += 2 * np.pi
            angular_velocity = 0.1 * angle_error

        return TurtlebotSteeringAction(
            linear_velocity=linear_velocity.item(),
            angular_velocity=angular_velocity.item(),
        )


class F1TenthSteeringAction(ControlAction):
    """The action for a F1Tenth steering controller."""

    steering_angle: float
    acceleration: float


class F1TenthSteeringPolicy(ControlPolicy):
    """Steer a F1Tenth towards a waypoint using an LQR controller.

    args:
        equilibrium_state: the state around which to linearize the dynamics
        axle_length: the distance between the front and rear axles
        dt: the time step for the controller
    """

    def __init__(self, equilibrium_state: np.ndarray, axle_length: float, dt: float):
        self.axle_length = axle_length
        self.dt = dt

        # Linearize the dynamics
        self.equilibrium_state = equilibrium_state
        A, B = self.get_AB(equilibrium_state, 0.0, 0.0)

        # Compute the LQR controller about the equilibrium
        Q = np.eye(4)
        R = np.eye(2)
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        self.K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    @property
    def observation_type(self):
        return SteeringObservation

    @property
    def action_type(self):
        return F1TenthSteeringAction

    def get_AB(self, state, delta, a):
        """
        Compute the linearized dynamics matrices.

        Args:
            state (np.ndarray): The current state [x, y, theta, v]
            delta (float): The steering angle command
            a (float): The acceleration command
        """
        # Extract the state variables
        _, _, theta, v = state

        # Compute the linearized dynamics matrices
        A = np.eye(4)
        A[0, 2] = -v * np.sin(theta) * self.dt
        A[0, 3] = np.cos(theta) * self.dt
        A[1, 2] = v * np.cos(theta) * self.dt
        A[1, 3] = np.sin(theta) * self.dt
        A[2, 3] = (1.0 / self.axle_length) * np.tan(delta) * self.dt

        B = np.zeros((4, 2))
        B[2, 0] = (v / self.axle_length) * self.dt / np.cos(delta) ** 2
        B[3, 1] = self.dt

        return A, B

    def compute_action(self, observation: SteeringObservation) -> F1TenthSteeringAction:
        """Takes in an observation and returns a control action."""
        state = np.array(
            [
                observation.pose.x,
                observation.pose.y,
                observation.pose.theta,
                self.equilibrium_state[3],
            ]
        ).reshape(-1, 1)
        goal = np.array(
            [
                observation.goal.x,
                observation.goal.y,
                observation.goal.theta,
                self.equilibrium_state[3],
            ]
        ).reshape(-1, 1)
        error = state - goal
        u = -self.K * error

        return F1TenthSteeringAction(
            steering_angle=u[0].item(), acceleration=u[1].item()
        )


if __name__ == "__main__":
    # Test the turtlebot steering policy
    policy = TurtlebotSteeringPolicy()

    initial_state = np.array([-1.0, -1.0, 1.0])
    states = [initial_state.tolist()]
    for i in range(500):
        action = policy.compute_action(
            SteeringObservation(
                pose=Pose2DObservation(
                    x=initial_state[0], y=initial_state[1], theta=initial_state[2]
                ),
                goal=Pose2DObservation(x=0.0, y=0.0, theta=0.0),
            )
        )
        initial_state += (
            np.array(
                [
                    action.linear_velocity * np.cos(initial_state[2]),
                    action.linear_velocity * np.sin(initial_state[2]),
                    action.angular_velocity,
                ]
            )
            * 0.05
        )
        states.append(initial_state.tolist())

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    states = np.array(states)
    plt.plot(states[:, 0], states[:, 1])
    plt.scatter(0.0, 0.0, c="r", label="Goal")
    plt.scatter(states[0, 0], states[0, 1], c="g", label="Start")
    plt.legend()
    plt.savefig("src/realm_gc/turtlebot_steering.png")
