"""Define policies used for TRO experiments."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from rgc_control.policies.composite import CompositePolicy
from rgc_control.policies.tracking.steering_policies import (
    F1TenthSteeringPolicy,
    TurtlebotSteeringPolicy,
)
from rgc_control.policies.tracking.tracking_policies import (
    TimedPose2DObservation,
    TrajectoryTrackingPolicy,
)
from rgc_control.policies.tracking.trajectory import LinearTrajectory2D
from rgc_control.policies.visual.common import VisualObservation
from rgc_control.policies.visual.vision_barrier_policies import (
    F1TenthVisionBarrierPolicy,
)
from rgc_control.policies.visual.vision_mlp_policies import F1TenthVisionMLPPolicy


@dataclass
class TROF1TenthObservation(TimedPose2DObservation, VisualObservation):
    """The observation type for the F1Tenth ego agent in the TRO experiment."""


@dataclass
class TROTurtlebotObservation(TimedPose2DObservation):
    """The observation type for the turtlebot nonego agents in the TRO experiment."""


def create_tro_f1tenth_policy(
    initial_state, traj_eqx_path, mlp_eqx_path
) -> CompositePolicy:
    """Create a composite policy for the F1Tenth ego agent in the TRO experiment.

    Args:
        initial_state: The initial 4D state of the F1Tenth.
        traj_eqx_path: The path to the trajectory (stored in an Equinox file).
        mlp_eqx_path: The path to the MLP (stored in an Equinox file).
    """
    # Construct the components of the policy using the parameters they were trained with

    # Start pointing along +y in the highbay
    desired_equilibrium_state = jnp.array([0.0, 0.0, jnp.pi / 2.0, 1.5])

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

    # Make the vision policy
    image_width = 16
    aspect = 4.0 / 3.0
    image_shape = (image_width, int(image_width / aspect))
    ego_mlp_policy = F1TenthVisionMLPPolicy.from_eqx(image_shape, mlp_eqx_path)

    # Make the barrier policy
    barrier_policy = F1TenthVisionBarrierPolicy(min_distance=1.0)

    # Combine the policies into a composite policy
    return CompositePolicy(
        [
            ego_tracking_policy,
            ego_mlp_policy,
            barrier_policy,
        ]
    )


def create_tro_turtlebot_policy(
    initial_position, traj_eqx_path, randomize=False
) -> CompositePolicy:
    """Create a composite policy for the turtlebot nonego agents in the TRO experiment.

    Args:
        initial_position: The initial 2D position of the turtlebot.
        traj_eqx_path: The path to the trajectory (stored in an Equinox file).
        randomize: Whether to randomize the trajectory according to the prior.
    """
    # Construct the components of the policy using the parameters they were trained with

    # Load the trajectory and flip the x and y coordinates, then add some noise
    non_ego_traj = LinearTrajectory2D.from_eqx(2, traj_eqx_path)
    p = jnp.fliplr(non_ego_traj.p)

    # # Clamp the initial position to be the intended starting position
    # if p[0, 1] <= -3.0:
    #     p = p.at[0, 0].set(-0.5)
    # else:
    #     p = p.at[0, 0].set(0.5)

    # Shift to +y to account for limited highbay space
    p = p.at[:, 1].add(0.5)

    # Upscale if it's small
    if p.shape == (2, 2):
        p_new = jnp.zeros((6, 2))
        p_new = p_new.at[:, 0].set(jnp.interp(jnp.linspace(0, 1, 6), jnp.array([0.0, 1.0]), p[:, 0]))
        p_new = p_new.at[:, 1].set(jnp.interp(jnp.linspace(0, 1, 6), jnp.array([0.0, 1.0]), p[:, 1]))
        p = p_new

    if randomize:
        noise_scale = 0.05
        p += np.random.normal(scale=np.sqrt(noise_scale), size=p.shape)

    non_ego_traj = LinearTrajectory2D(p=p)
    print("Loaded trajectory with waypoints:")
    print(non_ego_traj.p)

    # Make the trajectory tracking policy
    steering_controller = TurtlebotSteeringPolicy()
    ego_tracking_policy = TrajectoryTrackingPolicy(non_ego_traj, steering_controller)

    return ego_tracking_policy


if __name__ == "__main__":
    import matplotlib
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D

    matplotlib.use("Agg")

    # Load the policies
    ego_state = jnp.array([-5.5, -0.5, 0.0, 2.0 * 0.5])
    f1tenth_policy = create_tro_f1tenth_policy(
        ego_state,
        "/catkin_ws/src/realm_gc/rgc_control/saved_policies/base/ego_traj.eqx",
        "/catkin_ws/src/realm_gc/rgc_control/saved_policies/base/mlp.eqx",
    )

    non_ego_states = jnp.array([[-4.0, -0.5, 0.0], [-2.0, 0.5, 0.0]])
    non_ego_policies = [
        create_tro_turtlebot_policy(
            non_ego_states[i][:2],
            f"/catkin_ws/src/realm_gc/rgc_control/saved_policies/base/non_ego_traj_{i}.eqx",
        )
        for i in range(2)
    ]

    # Test the policies by simulating
    non_ego_states = [non_ego_state for non_ego_state in non_ego_states]

    ego_state_trace = []
    non_ego_states_trace = []

    steps = 100
    dt = 0.1

    for t in jnp.linspace(0, 1.0, steps):
        # Get the observations
        ego_obs = TROF1TenthObservation(
            x=ego_state[0],
            y=ego_state[1],
            theta=ego_state[2],
            v=ego_state[3],
            t=t,
            depth_image=jnp.zeros((16, 12)) + 10.0,
        )
        non_ego_obs = [
            TROTurtlebotObservation(
                x=non_ego_state[0],
                y=non_ego_state[1],
                theta=non_ego_state[2],
                v=0.0,  # not used for turtlebot
                t=t,
            )
            for non_ego_state in non_ego_states
        ]

        # Compute the actions
        ego_action = f1tenth_policy.compute_action(ego_obs)
        non_ego_actions = [
            non_ego_policy.compute_action(non_ego_obs)
            for non_ego_policy, non_ego_obs in zip(non_ego_policies, non_ego_obs)
        ]

        # Update the states
        ego_state = ego_state + dt * jnp.array(
            [
                ego_state[3] * jnp.cos(ego_state[2]),
                ego_state[3] * jnp.sin(ego_state[2]),
                ego_state[3] * ego_action.steering_angle / 0.28,
                ego_action.acceleration,
            ]
        )
        non_ego_states = [
            non_ego_state
            + dt
            * jnp.array(
                [
                    non_ego_action.linear_velocity * jnp.cos(non_ego_state[2]),
                    non_ego_action.linear_velocity * jnp.sin(non_ego_state[2]),
                    non_ego_action.angular_velocity,
                ]
            )
            for non_ego_state, non_ego_action in zip(non_ego_states, non_ego_actions)
        ]

        ego_state_trace.append(ego_state)
        non_ego_states_trace.append(non_ego_states)

    # Plot the results
    ego_state_trace = jnp.array(ego_state_trace)
    non_ego_states_trace = jnp.array(non_ego_states_trace)

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
    ax.plot(
        non_ego_states_trace[:, 0, 0],
        non_ego_states_trace[:, 0, 1],
        linestyle="-",
        color="blue",
        label="Actual trajectory (Non-ego 1)",
    )
    ax.plot(
        non_ego_states_trace[:, 1, 0],
        non_ego_states_trace[:, 1, 1],
        linestyle="-",
        color="blue",
        label="Actual trajectory (Non-ego 2)",
    )

    # Plot the trajectories
    ts = jnp.linspace(0, 1.0, steps)
    ego_planned_trajectory = jnp.array(
        [f1tenth_policy.policies[0].trajectory(t) for t in ts]
    )
    non_ego_planned_trajectory = jnp.array(
        [[policy.trajectory(t) for t in ts] for policy in non_ego_policies]
    )
    ax.plot(
        ego_planned_trajectory[:, 0],
        ego_planned_trajectory[:, 1],
        linestyle="--",
        color="red",
        label="Plan (Ego)",
    )
    ax.plot(
        non_ego_planned_trajectory[:, :, 0].T,
        non_ego_planned_trajectory[:, :, 1].T,
        linestyle="--",
        color="blue",
        label="Plan (Non-ego)",
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

    for i in [0, 1]:
        non_ego_car_pos = non_ego_states_trace[-1, i, :2]
        non_ego_car_heading = non_ego_states_trace[-1, i, 2]
        non_ego_car_patch = patches.Rectangle(
            (
                non_ego_car_pos[0] - car_length / 2,
                non_ego_car_pos[1] - car_width / 2,
            ),
            car_length,
            car_width,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        t = (
            Affine2D().rotate_deg_around(
                non_ego_car_pos[0],
                non_ego_car_pos[1],
                non_ego_car_heading * 180 / jnp.pi,
            )
            + ax.transData
        )
        non_ego_car_patch.set_transform(t)
        ax.add_patch(non_ego_car_patch)

    ax.legend()
    ax.set_aspect("equal")

    plt.savefig("src/realm_gc/tro_experiment.png")
