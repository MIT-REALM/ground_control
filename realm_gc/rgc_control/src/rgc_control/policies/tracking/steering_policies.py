"""Define policies for steering different dynamical systems towards waypoints."""
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.linalg as la
import math

from rgc_control.policies.common import F1TenthAction, TurtlebotAction
from rgc_control.policies.policy import ControlPolicy, Observation

from rgc_control.policies.tracking.trajectory import SplineTrajectory2D, LinearTrajectory2D

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



# utils from PythonRobotics

def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class F1TenthSpeedSteeringPolicy(ControlPolicy):
    """Steer a F1Tenth towards a waypoint using an LQR controller + control reference speed
    The implemented controller has been adopted from PythonRobotics repo:
    https://arxiv.org/abs/1808.10703

    args:
        equilibrium_state: the state around which to linearize the dynamics
        axle_length: the distance between the front and rear axles
        dt: the time step for the controller
    """

    def __init__(self, 
        trajectory: SplineTrajectory2D,
        # equilibrium_state: np.ndarray, 
        wheelbase: float, 
        dt: float, 
        target_speed=0.5):

        self.dt = dt

        # Get A,B matrices for this state
        # self.equilibrium_state = equilibrium_state
        # A, B = self.get_AB(equilibrium_state)

        # # Compute the LQR controller about the equilibrium
        # Q = np.eye(5) #Change
        # R = np.eye(2) #Change
        # X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        # self.K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

        self.lqr_Q = np.eye(5)
        self.lqr_R = np.eye(2)
        self.L = wheelbase  # Wheel base of the vehicle [m]

        self.target_speed = target_speed

        self.cx = trajectory.cx
        self.cy = trajectory.cy
        self.cyaw = trajectory.cyaw
        self.ck = trajectory.ck

        self.speed_profile = None
        self.set_calc_speed_profile()

    @property
    def observation_type(self):
        return SteeringObservation

    @property
    def action_type(self):
        return F1TenthAction

    def lqr_speed_steering_control(self, state):
        cx, cy, cyaw, ck = self.cx, self.cy, self.cyaw, self.ck
        pe, pth_e = state.e, state.theta_e
        sp, dt = self.speed_profile, self.dt
        L, Q, R = self.L, self.lqr_Q, self.lqr_R

        ind, e = self.calc_nearest_index(state)

        tv = sp[ind]

        k = ck[ind]
        v = state.v
        th_e = pi_2_pi(state.theta - cyaw[ind])

        # A = [1.0, dt, 0.0, 0.0, 0.0
        #      0.0, 0.0, v, 0.0, 0.0]
        #      0.0, 0.0, 1.0, dt, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = dt
        A[4, 4] = 1.0

        # B = [0.0, 0.0
        #     0.0, 0.0
        #     0.0, 0.0
        #     v/L, 0.0
        #     0.0, dt]
        B = np.zeros((5, 2))
        B[3, 0] = v / L
        B[4, 1] = dt

        K, _, _ = dlqr(A, B, Q, R)

        # state vector
        # x = [e, dot_e, th_e, dot_th_e, delta_v]
        # e: lateral distance to the path
        # dot_e: derivative of e
        # th_e: angle difference to the path
        # dot_th_e: derivative of th_e
        # delta_v: difference between current speed and target speed
        x = np.zeros((5, 1))
        x[0, 0] = e
        x[1, 0] = (e - pe) / dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / dt
        x[4, 0] = v - tv

        # input vector
        # u = [delta, accel]
        # delta: steering angle
        # accel: acceleration
        ustar = -K @ x

        # calc steering input
        ff = math.atan2(L * k, 1)  # feedforward steering angle
        fb = pi_2_pi(ustar[0, 0])  # feedback steering angle
        delta = ff + fb

        # calc accel input
        accel = ustar[1, 0]

        return delta, ind, e, th_e, accel

    def calc_nearest_index(self, state):
        cx, cy, cyaw = self.cx, self.cy, self.cyaw
        
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def set_calc_speed_profile(self):
        target_speed, cyaw = self.target_speed, self.cyaw

        speed_profile = [target_speed] * len(cyaw)

        direction = 1.0

        # Set stop point
        for i in range(len(cyaw) - 1):
            dyaw = abs(cyaw[i + 1] - cyaw[i])
            switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

            if switch:
                direction *= -1

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

            if switch:
                speed_profile[i] = 0.0

        # speed down
        for i in range(20):
            speed_profile[-i] = target_speed / (30 - i)
            if speed_profile[-i] <= 1.0 / 3.6:
                speed_profile[-i] = 1.0 / 3.6

        self.speed_profile = speed_profile
    
    # def get_AB(self,state):
    #     #Extract state variable here
    #     _,_,_, v = state
        
    #     A = np.zeros((5, 5))
    #     A[0, 0] = 1.0
    #     A[0, 1] = self.dt
    #     A[1, 2] = v
    #     A[2, 2] = 1.0
    #     A[2, 3] = self.dt
    #     A[4, 4] = 1.0

    #     B = np.zeros((5, 2))
    #     B[3, 0] = v / self.axle_length
    #     B[4, 1] = self.dt
    #     return A,B

    def compute_action(self, observation:SpeedSteeringObservation):
        state = observation.pose
        dl, target_ind, e, e_th, ai = self.lqr_speed_steering_control(state)
        return F1TenthAction(steering_angle=dl, acceleration=ai), e, e_th, target_ind

    
    # def compute_action(self,observation:SpeedSteeringObservation)-> F1TenthAction:
    #     state = np.array(
    #         [
    #             observation.pose.x,
    #             observation.pose.y,
    #             observation.pose.theta,
    #             observation.pose.v,
    #             observation.pose.e,
    #             observation.pose.theta_e,
    #         ]
    #     ).reshape(-1, 1)
    #     goal = np.array(
    #         [
    #             observation.goal.x,
    #             observation.goal.y,
    #             observation.goal.theta,
    #             observation.goal.v,
    #             observation.goal.k,
    #         ]
    #     ).reshape(-1, 1)

    #     """
    #     This block takes the nearest position to the current state as an input and returns curvilinear
    #     state error and computes corresponding control action to minimize that error
    #     """
    #     e = np.sqrt(np.power((goal[0]-state[0]),2) + np.power((goal[1]-state[1]),2))
    #     dxl = goal[0] - state[0]
    #     dyl = goal[1] - state[1]
    #     angle = self.pi_2_pi(goal[2] - np.arctan2(dyl, dxl))
    #     if angle < 0:
    #         e *= -1
    #     theta_e = self.pi_2_pi(goal[2]-state[2])

    #     x = np.zeros((5, 1))
    #     x[0, 0] = e
    #     x[1, 0] = (e - state[4]) / self.dt
    #     x[2, 0] = theta_e
    #     x[3, 0] = (theta_e - state[5]) / self.dt
    #     x[4, 0] = state[3] - goal[3]
    #     ustar = -self.K*x

    #     # calc steering itorchut
    #     ff = np.arctan2(np.array(self.axle_length * goal[4]), np.array(1))  # feedforward steering angle
    #     fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
    #     delta = -(fb +ff)

    #     # calc acceleration itorchut
    #     acc = ustar[1, 0]
    #     return F1TenthAction(steering_angle=delta.item(), acceleration=acc.item())
