"""Define a policy that is a composite of multiple policies."""
from dataclasses import fields
from typing import List

import numpy as np

from rgc_control.policies.common import ControlAction, Observation
from rgc_control.policies.policy import ControlPolicy


class CompositePolicy(ControlPolicy):
    """A policy that is a composite of multiple policies."""

    def __init__(self, policies: List[ControlPolicy]):
        self.policies = policies

    def compute_action(self, observation: Observation) -> ControlAction:
        """Compute the action by averaging the actions of the sub-policies."""
        actions = [policy.compute_action(observation) for policy in self.policies]

        # Raise an error if the policies return different action types
        action_types = {type(action) for action in actions}
        if len(action_types) > 1:
            raise ValueError(
                (
                    "CompositePolicy: policies returned "
                    f"different action types: {action_types}"
                )
            )

        # Create a new action by summing all fields of the sub-policy actions
        composite_action = type(actions[0])(
            **{
                field.name: np.sum([getattr(action, field.name) for action in actions])
                for field in fields(actions[0])
            }
        )

        return composite_action
