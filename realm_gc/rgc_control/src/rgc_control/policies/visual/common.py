from dataclasses import dataclass

import numpy as np

from rgc_control.policies.policy import Observation


@dataclass
class VisualObservation(Observation):
    """The observation for a single robot's visual input."""

    depth_image: np.ndarray
