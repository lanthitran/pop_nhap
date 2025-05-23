#!/usr/bin/env ipython

from dataclasses import dataclass
from pop.multiagent_system.reward_distributor import Incentivizer
from typing import Dict, Hashable, List

"""
This module implements a dictatorship penalization system for multi-agent reinforcement learning.
It helps prevent agents from getting stuck in repetitive patterns by applying penalties to repeated choices.

Key components:
- DictatorshipPenalizer: Main class that manages penalty calculations
- DictatorshipTracker: Inner class that tracks choice history and calculates penalties

The system uses exponential decay and growth functions to:
1. Scale penalties based on choice ranking
2. Increase penalties for repeated choices
3. Maintain minimum penalty thresholds

This helps encourage exploration and prevent agents from dominating with repetitive strategies.
| Hung |
"""

class DictatorshipPenalizer:
    """
    Main class that implements dictatorship penalization logic.
    
    This class:
    - Tracks agent choices and their rankings
    - Calculates penalties for repetitive behavior
    - Manages the penalty calculation parameters
    
    The penalty system discourages agents from repeatedly choosing the same actions
    by applying increasingly severe penalties for repetition.
    | Hung |
    """
    @dataclass
    class DictatorshipTracker:
        """
        Inner class that tracks individual choice history and calculates penalties.
        
        Attributes:
        - base_penalty_exponential_decay_half_life: Controls how quickly base penalty decays
        - penalty_exponential_growth_factor: Controls penalty growth rate for repetition
        - smallest_base_penalty: Minimum penalty threshold
        
        The tracker maintains state about:
        - Current choice and its rank
        - Number of repeated choices
        - Current base penalty value
        | Hung |
        """
        # TODO: here we assume that choices are non-negative and ranks too
        base_penalty_exponential_decay_half_life: float
        penalty_exponential_growth_factor: float
        smallest_base_penalty: float

        current_choice: int = -1
        repeated_choices: int = -1
        choice_rank: int = -1
        base_penalty: float = -1.0

        def choose(self, choice: int, choice_rank: int):
            """
            Updates tracker state based on new choice.
            
            If choice is repeated:
            - Increments repeated_choices counter
            Otherwise:
            - Resets repeated_choices
            - Updates current choice and rank
            - Recalculates base penalty
            | Hung |
            """
            if choice == self.current_choice:
                self.repeated_choices += 1
            else:
                self.current_choice = choice
                self.repeated_choices = 0
                self.choice_rank = choice_rank
                self.base_penalty = self._base_penalty(choice_rank)

        def dictatorship_penalty(self):
            """
            Calculates the current penalty based on:
            - Base penalty for the choice rank
            - Number of times choice was repeated
            - Exponential growth factor
            
            Returns negative value to be used as penalty
            | Hung |
            """
            return -Incentivizer._exponential_growth(
                self.base_penalty,
                self.penalty_exponential_growth_factor,
                self.repeated_choices,
            )

        def _base_penalty(self, choice_rank: int) -> float:
            """
            Calculates base penalty using exponential decay.
            
            Higher ranked choices get higher base penalties that decay
            according to the half-life parameter.
            | Hung |
            """
            return Incentivizer._exponential_decay(
                0,
                self.smallest_base_penalty,
                choice_rank,
                self.base_penalty_exponential_decay_half_life,
            )

        def reset(self):
            """
            Resets all tracker state to initial values.
            Used when starting new episodes or resetting the system.
            | Hung |
            """
            self.current_choice = -1
            self.repeated_choices = -1
            self.choice_rank = -1
            self.base_penalty = -1.0

    def __init__(
        self,
        choice_to_ranking: Dict[int, int],
        base_penalty_exponential_decay_half_life: float,
        penalty_exponential_growth_factor: float,
        smallest_base_penalty: float,
    ) -> None:
        """
        Initializes the penalizer with:
        - Mapping of choices to their rankings
        - Parameters controlling penalty calculations
        - A new dictatorship tracker instance
        | Hung |
        """
        self._choice_to_ranking = choice_to_ranking
        self._dictatorship_tracker = DictatorshipPenalizer.DictatorshipTracker(
            base_penalty_exponential_decay_half_life=base_penalty_exponential_decay_half_life,
            penalty_exponential_growth_factor=penalty_exponential_growth_factor,
            smallest_base_penalty=smallest_base_penalty,
        )

    def penalty(self, choice: int):
        """
        Calculates penalty for given choice by:
        1. Updating tracker with new choice
        2. Getting current penalty value
        
        Returns the calculated penalty value
        | Hung |
        """
        self._dictatorship_tracker.choose(choice, self._choice_to_ranking[choice])
        return self._dictatorship_tracker.dictatorship_penalty()

    def add_choice(self, choice: int, ranking: int):
        """
        Adds a new choice and its ranking to the mapping.
        Used to update available choices during runtime.
        | Hung |
        """
        self._choice_to_ranking[choice] = ranking

    def reset(self):
        """
        Resets the dictatorship tracker state.
        Called when starting new episodes or resetting the system.
        | Hung |
        """
        self._dictatorship_tracker.reset()
