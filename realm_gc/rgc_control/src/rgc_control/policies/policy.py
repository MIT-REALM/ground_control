"""Define generic control policy interface."""
from abc import ABC, abstractmethod

from rgc_control.policies.common import ControlAction, Observation


class ControlPolicy(ABC):
    @property
    def observation_type(self):
        return Observation

    @property
    def action_type(self):
        return ControlAction

    @abstractmethod
    def compute_action(self, observation: Observation) -> ControlAction:
        """Takes in an observation and returns a control action."""
        pass
