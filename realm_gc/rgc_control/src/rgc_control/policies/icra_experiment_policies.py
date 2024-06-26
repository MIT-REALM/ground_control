"""Define policies used for TRO experiments."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from rgc_control.policies.composite import CompositePolicy
from rgc_control.policies.tracking.steering_policies import ( F1TenthSteeringPolicy
)
from rgc_control.policies.tracking.tracking_policies import (
    TimedPose2DObservation,
    TrajectoryTrackingPolicy,
)
from rgc_control.policies.tracking.trajectory import LinearTrajectory2D

#Implements a trajectory-tracking LQR policy using an eqx filepath to give the waypoints via Bicycle dynamic model

@dataclass
class ICRAF1TenthObservation(TimedPose2DObservation):
    """The observation type for the turtlebot nonego agents in the TRO experiment."""

def create_icra_f1tenth_policy(
    initial_position, traj_eqx_path, randomize=False
) -> CompositePolicy:
    """Create a composite policy for the turtlebot nonego agents in the TRO experiment.

    Args:
        initial_position: The initial 2D position of the turtlebot.
        traj_eqx_path: The path to the trajectory (stored in an Equinox file).
        randomize: Whether to randomize the trajectory according to the prior.
    """
    # Construct the components of the policy using the parameters they were trained with

    # Start pointing along +y in the highbay
    desired_equilibrium_state = jnp.array([0.0, 0.0, jnp.pi / 2.0, 1.5])
    print("Instantiating LQR Tracking policy at [0,0,pi/2,1.5]")

    # Load the trajectory and flip x and y to convert from sim to high bay layout
    ego_traj = LinearTrajectory2D.from_eqx(6, traj_eqx_path)
    ego_traj = LinearTrajectory2D(p=jnp.fliplr(ego_traj.p))

    # Make the trajectory tracking policy
    steering_controller = F1TenthSteeringPolicy(
        equilibrium_state=desired_equilibrium_state,
        axle_length=0.28,
        dt=0.1,
    )
    ego_tracking_policy = TrajectoryTrackingPolicy(ego_traj, steering_controller)

    return ego_tracking_policy


if __name__ == "__main__":
    import matplotlib
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D

    matplotlib.use("Agg")

    # Load the policies
    ego_state = jnp.array([-5.5, -0.5, 0.0, 2.0 * 0.5])
    f1tenth_policy = create_icra_f1tenth_policy(
        ego_state,
        "/catkin_ws/src/realm_gc/rgc_control/saved_policies/base/ego_traj.eqx",)

    # Test the policies by simulating
    ego_state_trace = []

    steps = 100
    dt = 0.1

    for t in jnp.linspace(0, 1.0, steps):
        # Get the observations
        ego_obs = ICRAF1TenthObservation(
            x=ego_state[0],
            y=ego_state[1],
            theta=ego_state[2],
            v=ego_state[3],
            t=t,
        )
       
        # Compute the actions
        ego_action = f1tenth_policy.compute_action(ego_obs)
    
        # Update the states
        ego_state = ego_state + dt * jnp.array(
            [
                ego_state[3] * jnp.cos(ego_state[2]),
                ego_state[3] * jnp.sin(ego_state[2]),
                ego_state[3] * ego_action.steering_angle / 0.28,
                ego_action.acceleration,
            ]
        )
        
        ego_state_trace.append(ego_state)

    # Plot the results
    ego_state_trace = jnp.array(ego_state_trace)

    fig, ax = plt.subplots(1, 1, figsize=(32, 8))
    ax.axhline(-0.6, linestyle="--", color="k")
    ax.axhline(0.6, linestyle="--", color="k")
    ax.plot(
        ego_state_trace[:, 0].T,
        ego_state_trace[:, 1].T,
        linestyle="-",
        color="red",
        label="Actual trajectory (Ego)",
    )

    
    # Plot the trajectories
    ts = jnp.linspace(0, 1.0, steps)
    ego_planned_trajectory = jnp.array(
        [f1tenth_policy.policies[0].trajectory(t) for t in ts]
    )

    ax.plot(
        ego_planned_trajectory[:, 0],
        ego_planned_trajectory[:, 1],
        linestyle="--",
        color="red",
        label="Plan (Ego)",
    )
    
    # Draw a rectangular patch at the final car positions
    ego_car_pos = ego_state_trace[-1, :2]
    ego_car_heading = ego_state_trace[-1, 2]
    car_width = 0.28
    car_length = 0.3 + 0.04 + 0.09
    ego_car_patch = patches.Rectangle(
        (ego_car_pos[0] - car_length / 2, ego_car_pos[1] - car_width / 2),
        car_length,
        car_width,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    t = (
        Affine2D().rotate_deg_around(
            ego_car_pos[0], ego_car_pos[1], ego_car_heading * 180 / jnp.pi
        )
        + ax.transData
    )
    ego_car_patch.set_transform(t)
    ax.add_patch(ego_car_patch)

    ax.legend()
    ax.set_aspect("equal")

    plt.savefig("src/realm_gc/f1tenth_lqr_experiment.png")
