"""Define policies that use MLPs with vision inputs."""
import equinox as eqx
import jax
import jax.numpy as jnp

from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.policy import ControlPolicy
from rgc_control.policies.visual.common import VisualObservation


class F1TenthVisionMLPPolicy(ControlPolicy):
    """Steers the F1tenth using an MLP with visual inputs."""

    def __init__(self, key, image_shape):
        """Initialize the policy."""
        super().__init__()
        self.actor_fcn = eqx.nn.MLP(
            in_size=image_shape[0] * image_shape[1],
            out_size=2,
            width_size=32,
            depth=3,
            key=key,
            activation=jax.nn.tanh,
        )

    @staticmethod
    def from_eqx(image_shape, filepath):
        """Load a F1TenthVisionMLPPolicy from a file."""
        key = jax.random.PRNGKey(0)
        policy = F1TenthVisionMLPPolicy(key, image_shape)
        policy.actor_fcn = eqx.tree_deserialise_leaves(filepath, policy.actor_fcn)
        return policy

    def compute_action(self, obs: VisualObservation) -> F1TenthAction:
        """Compute the action for the given observation using the MLP."""
        depth_image = obs.depth_image
        min_distance = 1.0
        depth_image = jnp.where(depth_image < 1e-3, min_distance, depth_image)
        image = jnp.reshape(depth_image, (-1,))
        action = 0.1 * self.actor_fcn(image)

        action = F1TenthAction(
            acceleration=action[0],
            steering_angle=-action[1],
        )

        return action
