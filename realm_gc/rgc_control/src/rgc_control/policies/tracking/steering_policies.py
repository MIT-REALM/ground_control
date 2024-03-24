"""Define policies for steering different dynamical systems towards waypoints."""
from dataclasses import dataclass

import numpy as np
import scipy
from rgc_control.policies.common import F1TenthAction, TurtlebotAction
from rgc_control.policies.policy import ControlPolicy, Observation

@dataclass
class Pose2DObservation(Observation):
    """The observation for a single robot's 2D pose and linear speed."""

    x: float
    y: float
    theta: float
    v: float

@dataclass
class G2CPose2DObservation(Observation):
    """The observation for a single robot's 2D pose and linear speed."""

    x: float
    y: float
    theta: float
    v: float
    e: float
    theta_e: float

@dataclass
class G2CGoal2DObservation(Observation):
    """The observation for a single robot's 2D pose and linear speed."""

    x: float
    y: float
    theta: float
    v:float
    k: float

@dataclass
class SteeringObservation(Observation):
    """The observation for a single robot's 2D pose relative to some goal."""

    pose: Pose2DObservation
    goal: Pose2DObservation

@dataclass
class SpeedSteeringObservation(Observation):
    """This class computes corresponding terms for Speed+Steer control ."""

    pose: G2CPose2DObservation
    goal: G2CGoal2DObservation


class TurtlebotSteeringPolicy(ControlPolicy):
    """Steer a turtlebot towards a waypoint using a proportional controller."""

    @property
    def observation_type(self):
        return SteeringObservation

    @property
    def action_type(self):
        return TurtlebotAction

    def compute_action(self, observation: SteeringObservation) -> TurtlebotAction:
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
        linear_velocity = 1.0 * error[0]  # projection along the turtlebot x-axis

        # Compute the angular velocity: steer towards the goal if we're far from it
        # (so the arctan is well defined), and align to the goal orientation
        if np.linalg.norm(error) > 0.1:
            angular_velocity = np.arctan2(error[1], error[0])
        else:
            angle_error = observation.goal.theta - observation.pose.theta
            if angle_error > np.pi:
                angle_error -= 2 * np.pi
            if angle_error < -np.pi:
                angle_error += 2 * np.pi
            angular_velocity = 1.0 * angle_error

        if isinstance(linear_velocity, np.ndarray):
            linear_velocity = linear_velocity.item()

        if isinstance(angular_velocity, np.ndarray):
            angular_velocity = angular_velocity.item()

        return TurtlebotAction(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )


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
        return F1TenthAction

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

    def compute_action(self, observation: SteeringObservation) -> F1TenthAction:
        """Takes in an observation and returns a control action."""
        state = np.array(
            [
                observation.pose.x,
                observation.pose.y,
                observation.pose.theta,
                observation.pose.v,
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

        return F1TenthAction(steering_angle=u[0].item(), acceleration=u[1].item())



class F1TenthSpeedSteeringPolicy(ControlPolicy):
    """Steer a F1Tenth towards a waypoint using an LQR controller + control reference speed
    The implemented controller has been adopted from PythonRobotics repo:
    https://arxiv.org/abs/1808.10703

    args:
        equilibrium_state: the state around which to linearize the dynamics
        axle_length: the distance between the front and rear axles
        dt: the time step for the controller
    """

    def __init__(self, equilibrium_state: np.ndarray, axle_length: float, dt: float):
        self.axle_length = axle_length
        self.dt = dt

        # Get A,B matrices for this state
        self.equilibrium_state = equilibrium_state
        A, B = self.get_AB(equilibrium_state)

        # Compute the LQR controller about the equilibrium
        Q = np.eye(5) #Change
        R = np.eye(2) #Change
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        self.K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    @property
    def observation_type(self):
        return SteeringObservation

    @property
    def action_type(self):
        return F1TenthAction

    def get_AB(self,state):
        #Extract state variable here
        _,_,_, v = state
        
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = self.dt
        A[4, 4] = 1.0

        B = np.zeros((5, 2))
        B[3, 0] = v / self.axle_length
        B[4, 1] = self.dt
        return A,B
    
    def pi_2_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def compute_action(self,observation:SpeedSteeringObservation)-> F1TenthAction:
        state = np.array(
            [
                observation.pose.x,
                observation.pose.y,
                observation.pose.theta,
                observation.pose.v,
                observation.pose.e,
                observation.pose.theta_e,
            ]
        ).reshape(-1, 1)
        goal = np.array(
            [
                observation.goal.x,
                observation.goal.y,
                observation.goal.theta,
                observation.goal.v,
                observation.goal.k,
            ]
        ).reshape(-1, 1)

        """
        This block takes the nearest position to the current state as an input and returns curvilinear
        state error and computes corresponding control action to minimize that error
        """
        e = np.sqrt(np.power((goal[0]-state[0]),2) + np.power((goal[1]-state[1]),2))
        dxl = goal[0] - state[0]
        dyl = goal[1] - state[1]
        angle = self.pi_2_pi(goal[2] - np.arctan2(dyl, dxl))
        if angle < 0:
            e *= -1
        theta_e = self.pi_2_pi(goal[2]-state[2])

        x = np.zeros((5, 1))
        x[0, 0] = e
        x[1, 0] = (e - state[4]) / self.dt
        x[2, 0] = theta_e
        x[3, 0] = (theta_e - state[5]) / self.dt
        x[4, 0] = state[3] - goal[3]
        ustar = -self.K*x

        # calc steering itorchut
        ff = np.arctan2(np.array(self.axle_length * goal[4]), np.array(1))  # feedforward steering angle
        fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
        delta = -(fb +ff)

        # calc acceleration itorchut
        acc = ustar[1, 0]
        return F1TenthAction(steering_angle=delta.item(), acceleration=acc.item())
