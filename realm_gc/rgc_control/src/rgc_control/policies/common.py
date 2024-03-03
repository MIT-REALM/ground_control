"""Define commonly used action types."""
from dataclasses import dataclass


@dataclass
class Observation:
    pass


@dataclass
class ControlAction:
    pass


@dataclass
class TurtlebotAction(ControlAction):
    """The action for a turtlebot steering controller."""

    linear_velocity: float
    angular_velocity: float


@dataclass
class F1TenthAction(ControlAction):
    """The action for a F1Tenth steering controller."""

    steering_angle: float
    acceleration: float
