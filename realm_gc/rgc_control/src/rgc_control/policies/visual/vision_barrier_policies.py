"""Define policies that avoid obstacles using vision."""

import jax.numpy as jnp

from rgc_control.policies.common import F1TenthAction
from rgc_control.policies.policy import ControlPolicy
from rgc_control.policies.visual.common import VisualObservation


class F1TenthVisionBarrierPolicy(ControlPolicy):
    """Brakes as needed to avoid obstacles."""

    def __init__(self, min_distance: float = 1.0):
        self.min_distance = min_distance

    def gaussian_kernel(self, shape, sigma=1.0):
        """Generates a 2D Gaussian kernel."""
        ay = jnp.arange(-shape[0] // 2, shape[0] // 2)
        ax = jnp.arange(-shape[1] // 2, shape[1] // 2)
        xx, yy = jnp.meshgrid(ax, ay)
        kernel = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / jnp.sum(kernel)

    def compute_action(
        self,
        obs: VisualObservation,
    ) -> F1TenthAction:
        # Brake to avoid collisions based on the average distance to the
        # obstacle in the center of the image
        depth_image = obs.depth_image
        depth_image = jnp.where(depth_image < 1e-3, self.min_distance, depth_image)
        kernel = self.gaussian_kernel(depth_image.shape, 1.0)
        mean_distance = jnp.sum(depth_image * kernel) / jnp.sum(kernel)
        accel = 5 * jnp.clip(mean_distance - self.min_distance, -2.0, 0.0)
        print(f"Vision barrier: mean_distance: {mean_distance},  accel {accel}")

        return F1TenthAction(
            acceleration=accel,
            steering_angle=0.0,
        )
