"""Define linearly interpolated trajectories."""
from typing import List
import pickle
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, Float
from rgc_control.policies.cubic_spline import CubicSpline2D

class LinearTrajectory2D(eqx.Module):
    """
    The trajectory for a single robot, represented by linear interpolation.

    Time is normalized to [0, 1]

    args:
        p: the array of control points for the trajectory
    """

    p: Float[Array, "T 2"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "2"]:
        """Return the point along the trajectory at the given time"""
        # Interpolate each axis separately
        return jnp.array(
            [
                jnp.interp(
                    t,
                    jnp.linspace(0, 1, self.p.shape[0]),
                    self.p[:, i],
                )
                for i in range(2)
            ]
        )

    @staticmethod
    def from_eqx(T: int, filepath: str) -> "LinearTrajectory2D":
        """Load a LinearTrajectory2D from a file."""
        traj = LinearTrajectory2D(jnp.zeros((T, 2)))
        traj = eqx.tree_deserialise_leaves(filepath, traj)
        return traj

class SplineTrajectory2D():
    """
    The trajectory for a single robot, represented by curvilinear interpolation
    t represents the index in the spline list
    args:
        p: the array of control points for the trajectory
    """
    def __init__(self, v_ref:float, filepath: str):
        #Loads a dictionary with keys 'X' and 'Y' and converts it into spline information
        file = open(filepath,'rb')
        self.traj = pickle.load(file) 
        file.close()
        self.cx,self.cy,self.cyaw,self.ck = self.calc_spline_course(self.traj)
        self.v_ref = v_ref
        self.v = self.calc_speed_profile(self.v_ref)

    def calc_spline_course(self,trajectory, ds=0.1):
        x = trajectory['X']
        y = trajectory['Y']
        sp = CubicSpline2D(x, y)
        if np.isnan(sp.s[-1]):
            print(sp.s)
        s = list(np.arange(0, sp.s[-1], ds))

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))
        return rx, ry, ryaw, rk
    
    def calc_speed_profile(self,v_ref):
        speed_profile = [v_ref] * len(self.cyaw)
        direction = 1.0

        # Set stop point
        for i in range(len(self.cyaw) - 1):
            dyaw = abs(self.cyaw[i + 1] - self.cyaw[i])
            switch = torch.pi / 4.0 <= dyaw < torch.pi / 2.0

            if switch:
                direction *= -1

            if direction != 1.0:
                speed_profile[i] = - v_ref
            else:
                speed_profile[i] = v_ref

            if switch:
                speed_profile[i] = 0.0

        # speed down
        if i>20:
            for i in range(20):
                speed_profile[-i] = v_ref / (50 - i)
                if speed_profile[-i] <= 1.0 / 3.6:
                    speed_profile[-i] = 1.0 / 3.6
            return speed_profile

    def __call__(self, t: int) -> Float[Array, "5"]:
        """Return the point along the trajectory at the given index"""
        return torch.tensor([self.cx[t],self.cy[t], self.cyaw[t], self.v[t], self.ck[t]])

    


class MultiAgentTrajectoryLinear(eqx.Module):
    """
    The trajectory for a swarm of robots.

    args:
        trajectories: the list of trajectories for each robot.
    """

    trajectories: List[LinearTrajectory2D]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "N 2"]:
        """Return the waypoints for each agent at a given time (linear interpolate)"""
        return jnp.array([traj(t) for traj in self.trajectories])

    @staticmethod
    def from_eqx(N: int, T: int, filepath: str) -> "MultiAgentTrajectoryLinear":
        """Load a MultiAgentTrajectoryLinear from a file."""
        trajs = [LinearTrajectory2D(jnp.zeros((T, 2))) for _ in range(N)]
        trajs = eqx.tree_deserialise_leaves(filepath, trajs)
        return trajs
