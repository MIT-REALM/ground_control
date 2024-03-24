"""Define policies used for TRO experiments."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from rgc_control.policies.composite import CompositePolicy
from rgc_control.policies.tracking.steering_policies import (
    F1TenthSpeedSteeringPolicy,
)
from rgc_control.policies.tracking.tracking_policies import (
    TimedG2CPose2DObservation,
    G2CTrajectoryTrackingPolicy,
)
from rgc_control.policies.tracking.trajectory import SplineTrajectory2D


@dataclass
class RALF1tenthObservation(TimedG2CPose2DObservation):
    """The observation type for the F1 tenth ego agent in the RAL experiment."""

def create_ral_f1tenth_policy(
    initial_position, v_ref, traj_csv_path
) -> CompositePolicy:
    """Create a composite policy for the F1tenth ego agent in the RAL experiment.

    Args:
        initial_position: The initial 2D position of F1 tenth.
        traj_csv_path: The path to the trajectory (stored in an CSV file).
    """
    # Construct the components of the policy using the parameters they were trained with

    # Load the trajectory and flip the x and y coordinates, then add some noise
    ego_traj = SplineTrajectory2D(v_ref, traj_csv_path)

    # Start pointing along +y in the highbay
    desired_equilibrium_state = jnp.array([0.0, 0.0, jnp.pi / 2.0, 1.5])

    #p = jnp.fliplr(ego_traj.p) 

    # # Clamp the initial position to be the intended starting position
    # if p[0, 1] <= -3.0:
    #     p = p.at[0, 0].set(-0.5)
    # else:
    #     p = p.at[0, 0].set(0.5)

    # Shift to +y to account for limited highbay space
    #p = p.at[:, 1].add(0.5)

    # Upscale if it's small
    #if p.shape == (2, 2):
    #    p_new = jnp.zeros((6, 2))
    #    p_new = p_new.at[:, 0].set(jnp.interp(jnp.linspace(0, 1, 6), jnp.array([0.0, 1.0]), p[:, 0]))
    #    p_new = p_new.at[:, 1].set(jnp.interp(jnp.linspace(0, 1, 6), jnp.array([0.0, 1.0]), p[:, 1]))
    #    p = p_new

    print("Loaded trajectory with waypoints:")

    print(ego_traj)

    # Make the trajectory tracking policy
    steering_controller = F1TenthSpeedSteeringPolicy(
        equilibrium_state=desired_equilibrium_state,
        axle_length=0.28,
        dt=0.03,
    )
    ego_tracking_policy = G2CTrajectoryTrackingPolicy(ego_traj, steering_controller)

    return ego_tracking_policy


if __name__ == "__main__":
    import matplotlib
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D

    matplotlib.use("Agg")

    # Load the policies
    ego_state = jnp.array([-5.5, -0.5, 0.0, 2.0 * 0.5,0.0,0.0])
    f1tenth_policy = create_ral_f1tenth_policy(
        ego_state,
        "/catkin_ws/src/realm_gc/rgc_control/saved_policies/base/ego_traj.pkl",
    )

    # Test the policies by simulating
    ego_state_trace = []

    steps = 100
    dt = 0.1

    for t in jnp.linspace(0, 1.0, steps):
        # Get the observations
        ego_obs = RALF1tenthObservation(
            x=ego_state[0],
            y=ego_state[1],
            theta=ego_state[2],
            v=ego_state[3],
            e = ego_state[4],
            theta_e = ego_state[5],
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

    plt.savefig("src/realm_gc/ral_experiment.png")
