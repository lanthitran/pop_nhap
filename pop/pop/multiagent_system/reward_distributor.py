from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

from math import exp, log10

"""
The Incentivizer class manages reward distribution in a multi-agent system by tracking agent performance
and applying dynamic incentives based on election outcomes. It uses a combination of exponential and
logarithmic growth functions to calculate rewards and penalties.

Key components:
- ElectionHistory: Tracks consecutive elections and rejections for each agent
- Incentive: Calculates rewards/penalties based on agent performance history
- Dynamic ranking system based on agent action spaces
| Hung |
"""

class Incentivizer:
    """
    Manages incentive distribution for agents based on their election performance.
    Uses exponential and logarithmic growth functions to calculate rewards/penalties.
    | Hung |
    """
    @dataclass
    class ElectionHistory:
        """
        Tracks election history for individual agents including consecutive
        elections and rejections.
        | Hung |
        """
        consecutive_elections: int = 0
        consecutive_rejections: int = 0

        def elect(self):
            # Increment elections counter and reset rejections | Hung |
            self.consecutive_elections += 1
            self.consecutive_rejections = 0

        def reject(self):
            # Increment rejections counter and reset elections | Hung |
            self.consecutive_rejections += 1
            self.consecutive_elections = 0

        def reset(self):
            # Reset all counters to zero | Hung |
            self.consecutive_elections = 0
            self.consecutive_rejections = 0

    @dataclass
    class Incentive:
        """
        Calculates rewards and penalties based on agent performance history.
        Uses exponential growth for penalties and logarithmic growth for rewards.
        | Hung |
        """
        minimum_penalty: float
        minimum_prize: float
        penalty_growth_rate: float
        prize_growth_rate: float

        def incentive(self, election_history: "Incentivizer.ElectionHistory") -> float:
            # Return prize for consecutive elections or penalty for rejections | Hung |
            if election_history.consecutive_elections > 0:
                return self._prize(election_history.consecutive_elections)
            return self._penalty(election_history.consecutive_rejections)

        def _penalty(self, consecutive_rejections: int):
            # Calculate penalty using exponential growth | Hung |
            return -Incentivizer._exponential_growth(
                self.minimum_penalty,
                self.penalty_growth_rate,
                consecutive_rejections - 1,
            )

        def _prize(self, consecutive_elections: int):
            # Calculate prize using logarithmic growth | Hung |
            return Incentivizer._logarithmic_growth(
                self.minimum_prize, self.prize_growth_rate, consecutive_elections - 1
            )

    def __init__(
        self,
        agent_actions: Dict[Hashable, int],
        largest_base_prize: float,
        smallest_base_penalty: float,
        prize_logarithmic_growth_factor: float,
        penalty_exponential_growth_factor: float,
        base_prize_exponential_decay_half_life: float,
        base_penalty_exponential_growth_factor: float,
    ) -> None:
        # State does not need to be loaded nor saved
        # Elections are reset at every episode

        # Initialize agent tracking and ranking | Hung |
        self._agents: List[Hashable] = list(agent_actions.keys())
        self.agent_actions = agent_actions
        self._rank_to_agents: Dict[int, List[Hashable]] = self._rank_agents(
            self.agent_actions
        )
        self._elections: Dict[Hashable, Incentivizer.ElectionHistory] = {
            agent: Incentivizer.ElectionHistory() for agent in self._agents
        }

        # Store incentive calculation parameters | Hung |
        self._smallest_penalty: float = smallest_base_penalty
        self._largest_prize: float = largest_base_prize
        self._prize_logarithmic_growth_factor: float = prize_logarithmic_growth_factor
        self._penalty_exponential_growth_factor: float = (
            penalty_exponential_growth_factor
        )
        self._base_prize_exponential_decay_factor: float = (
            base_prize_exponential_decay_half_life
        )
        self._base_penalty_exponential_growth_factor: float = (
            base_penalty_exponential_growth_factor
        )

        # Calculate initial incentives for all agents | Hung |
        self._agents_incentives: Dict[
            Hashable, "Incentivizer.Incentive"
        ] = self._compute_base_incentives(
            agent_ranking=self._rank_to_agents,
            smallest_penalty=self._smallest_penalty,
            largest_prize=self._largest_prize,
            prize_exponential_decay_factor=base_prize_exponential_decay_half_life,
            penalty_exponential_growth_factor=base_penalty_exponential_growth_factor,
        )

    def incentives(
        self,
        elected_agents: List[Hashable],
    ) -> Dict[Hashable, float]:
        """
        Calculate incentives for all agents based on election results.
        Updates election history and returns incentive values.
        | Hung |
        """
        # Update election history for all agents | Hung |
        for agent in self._agents:
            if agent in elected_agents:
                self._elections[agent].elect()
            else:
                self._elections[agent].reject()

        # Calculate and return incentives for each agent | Hung |
        return {
            agent: self._agents_incentives[agent].incentive(self._elections[agent])
            for agent in self._agents
        }

    def add_agent(self, agent_to_add: Hashable, actions: int):
        """
        Add a new agent to the incentive system and recalculate rankings.
        | Hung |
        """
        # Add agent and update rankings | Hung |
        self._agents.append(agent_to_add)
        self.agent_actions[agent_to_add] = actions
        self._rank_to_agents = self._rank_agents(self.agent_actions)
        self._elections = {
            agent: Incentivizer.ElectionHistory() for agent in self._agents
        }
        # Recalculate incentives for all agents | Hung |
        self._agents_incentives: Dict[
            Hashable, "Incentivizer.Incentive"
        ] = self._compute_base_incentives(
            agent_ranking=self._rank_to_agents,
            smallest_penalty=self._smallest_penalty,
            largest_prize=self._largest_prize,
            prize_exponential_decay_factor=self._base_prize_exponential_decay_factor,
            penalty_exponential_growth_factor=self._base_penalty_exponential_growth_factor,
        )

    def reset(self):
        """
        Reset election history for all agents.
        | Hung |
        """
        for election_history in self._elections.values():
            election_history.reset()

    def _invert_ranking(
        self, ranking: Dict[int, List[Hashable]]
    ) -> Dict[Hashable, List[int]]:
        """
        Convert ranking dictionary from rank->agents to agent->rank.
        | Hung |
        """
        return {
            agent: rank for rank, agents in ranking.items() for agent in agents
        }  # type: ignore

    def _compute_base_incentives(
        self,
        agent_ranking: Dict[int, List[Hashable]],
        largest_prize: float,
        smallest_penalty: float,
        prize_exponential_decay_factor: float,
        penalty_exponential_growth_factor: float,
    ) -> Dict[Hashable, "Incentivizer.Incentive"]:
        """
        Calculate base incentives for all agents based on their ranking.
        Uses exponential decay for prizes and exponential growth for penalties.
        | Hung |
        """
        incentives: Dict[Hashable, "Incentivizer.Incentive"] = {}
        for rank, agents in agent_ranking.items():
            # Calculate base penalty and prize for this rank | Hung |
            base_penalty: float = Incentivizer._exponential_growth(
                smallest_penalty, penalty_exponential_growth_factor, rank
            )
            base_prize: float = Incentivizer._exponential_decay(
                0, largest_prize, rank, prize_exponential_decay_factor
            )
            # Create incentive object for each agent at this rank | Hung |
            for agent in agents:
                incentives[agent] = Incentivizer.Incentive(
                    minimum_penalty=base_penalty,
                    minimum_prize=base_prize,
                    penalty_growth_rate=self._penalty_exponential_growth_factor,
                    prize_growth_rate=self._prize_logarithmic_growth_factor,
                )
        return incentives

    @staticmethod
    def _rank_agents(agent_actions: Dict[Hashable, int]) -> Dict[int, List[Hashable]]:
        """
        Rank agents based on their action space size relative to total actions.
        | Hung |
        """
        full_action_space = sum(list(agent_actions.values()))
        ranking: Dict[int, List[Hashable]] = {}
        ordered_action_spaces: Dict[float, List[Hashable]] = {}
        
        # Calculate action space portions for each agent | Hung |
        for agent, action_space_size in agent_actions.items():
            action_space_portion = full_action_space / action_space_size
            if ordered_action_spaces.get(action_space_portion):
                ordered_action_spaces[action_space_portion].append(agent)
            else:
                ordered_action_spaces[action_space_portion] = [agent]

        ordered_action_spaces = dict(
            sorted(ordered_action_spaces.items(), reverse=True)
        )
        for position, (action_space_portion, agents) in enumerate(
            ordered_action_spaces.items()
        ):
            ranking[position] = agents

        return ranking

    @staticmethod
    def _exponential_decay(
        initial_value: float, target_value: float, steps: int, half_life: float
    ) -> float:
        return initial_value + (target_value - initial_value) * exp(
            -1.0 * steps / half_life
        )

    @staticmethod
    def _exponential_growth(
        initial_value: float, growth_rate: float, steps: int
    ) -> float:
        return initial_value * (1 + growth_rate) ** steps

    @staticmethod
    def _logarithmic_growth(
        initial_value: float, growth_rate: float, steps: int
    ) -> float:
        return growth_rate * log10(steps + 1) + initial_value
