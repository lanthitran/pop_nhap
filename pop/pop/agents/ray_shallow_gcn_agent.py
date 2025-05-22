from typing import Optional, Dict, Any, List, Tuple

import networkx as nx
from dgl import DGLHeteroGraph
from ray import ObjectRef

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture
"""
RayShallowGCNAgent is a simplified version of the GCN agent that implements a 
minimal interface for distributed computing using Ray. It serves as a lightweight 
agent that always returns no-action, useful for testing and baseline comparisons 
in the distributed reinforcement learning system.
| Hung |
"""

class RayShallowGCNAgent(BaseGCNAgent):
    """
    A minimal implementation of a distributed GCN agent that always returns 
    no-action. Inherits from BaseGCNAgent and provides basic functionality 
    for distributed computing with Ray.
    | Hung |
    """
    def __init__(
        self,
        name: str,
        device: str,
        agent_actions: Optional[int] = None,
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
        architecture: Optional[AgentArchitecture] = None,
        training: bool = False,
    ):
        """
        Initialize the shallow GCN agent with basic configuration.
        Only stores name and device information for minimal functionality.
        | Hung |
        """
        self.name = name
        self.device = device

    def get_state(
        self,
    ) -> Dict[str, Any]:
        """
        Returns a minimal state dictionary containing only name and device info.
        Used for checkpointing and state management in distributed computing.
        | Hung |
        """
        return {
            "name": self.name,
            "device": self.device,
        }

    def get_exploration_logs(self) -> Dict[str, Any]:
        """
        Returns an empty dictionary as this agent doesn't perform exploration.
        Maintains interface compatibility with other GCN agents.
        | Hung |
        """
        return {}

    def get_name(self) -> str:
        """
        Returns the agent's name identifier.
        Used for logging and identification in distributed systems.
        | Hung |
        """
        return self.name

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs):
        """
        Creates a new shallow GCN agent instance from a checkpoint.
        Only restores basic configuration without complex state.
        | Hung |
        """
        agent = RayShallowGCNAgent(
            name=checkpoint["name"], 
            device=checkpoint["device"]
        )
        return agent

    def take_action(
        self, transformed_observation: DGLHeteroGraph, mask: List[int] = None
    ) -> Tuple[int, float]:
        """
        Always returns no-action (0) with zero Q-value.
        Implements minimal action selection for baseline comparisons.
        | Hung |
        """
        return 0, 0  # Always no-action with 0 q-value

    def step(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: nx.Graph,
        done: bool,
        stop_decay: bool = False,
    ) -> Tuple[None, float]:
        """
        Performs a minimal step that only returns the reward.
        No learning or state updates are performed.
        | Hung |
        """
        return None, reward
