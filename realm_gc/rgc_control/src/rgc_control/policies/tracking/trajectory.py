"""Define linearly interpolated trajectories."""
from typing import List

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


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
