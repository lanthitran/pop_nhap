from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import toml

from pop.configs.agent_architecture import AgentArchitecture
from pop.configs.type_aliases import ParsedTOMLDict
"""
This module defines the core architecture configuration classes for the POP (Power Optimization Platform) system.
It provides data structures and initialization logic for managing different components of the power system architecture.
"""

@dataclass(frozen=True)
class POPArchitecture:
    """
    Configuration class for the overall POP system architecture.
    
    This class defines the core parameters and features for the power system optimization platform,
    including node and edge features, agent behaviors, and various system constraints.
    
    Attributes:
        node_features: List of features associated with each node in the power network
        edge_features: List of features associated with each edge in the power network
        agent_neighbourhood_radius: Radius within which agents can interact
        head_manager_embedding_name: Name of the embedding used for community actions
        decentralized: Whether the system operates in decentralized mode
        epsilon_beta_scheduling: Whether to use epsilon-beta scheduling
        enable_power_supply_modularity: Whether to enable modular power supply features
        manager_history_size: Size of the manager's history buffer
        manager_initialization_half_life: Half-life for manager initialization
        agent_type: Type of agent implementation ("uniform" by default)
        disabled_action_loops_length: Length of disabled action loops
        repeated_action_penalty: Penalty for repeated actions
        manager_selective_learning: Whether manager uses selective learning
        agent_selective_learning: Whether agents use selective learning
        composite_actions: Whether to enable composite actions
        no_action_reward: Whether to include rewards for no-action
        incentives: Optional dictionary of incentive configurations
        dictatorship_penalty: Optional dictionary of dictatorship penalty configurations
        enable_expert: Whether to enable expert mode
        safe_max_rho: Maximum safe rho value
        curtail_storage_limit: Storage limit for curtailment
        actions_per_generator: Number of actions per generator
        generator_storage_only: Whether generators are storage-only
        remove_no_action: Whether to remove no-action option
        manager_remove_no_action: Whether manager should remove no-action option
    """
    node_features: List[str]
    edge_features: List[str]
    agent_neighbourhood_radius: int = 1
    head_manager_embedding_name: str = "embedding_community_action"
    decentralized: bool = False
    epsilon_beta_scheduling: bool = False
    enable_power_supply_modularity: bool = False
    manager_history_size: int = int(1e5)
    manager_initialization_half_life: int = 0
    agent_type: str = "uniform"
    disabled_action_loops_length: int = 0
    repeated_action_penalty: float = 0
    manager_selective_learning: bool = False
    agent_selective_learning: bool = False
    composite_actions: bool = False
    no_action_reward: bool = False
    incentives: Optional[Dict[str, Any]] = None
    dictatorship_penalty: Optional[Dict[str, Any]] = None
    enable_expert: bool = False
    safe_max_rho: float = 0.99
    curtail_storage_limit: float = 10
    actions_per_generator: int = 10
    generator_storage_only: bool = False
    remove_no_action: bool = False
    manager_remove_no_action: bool = False


@dataclass(frozen=True)
class Architecture:
    """
    Main architecture class that combines all components of the POP system.
    
    This class serves as the top-level configuration container, bringing together
    the POP architecture, agent, manager, and head manager configurations.
    
    Attributes:
        pop: POPArchitecture instance for overall system configuration
        agent: AgentArchitecture instance for agent configuration
        manager: AgentArchitecture instance for manager configuration
        head_manager: AgentArchitecture instance for head manager configuration
    """
    pop: POPArchitecture
    
    agent: AgentArchitecture

    manager: AgentArchitecture

    head_manager: AgentArchitecture

    def __init__(
        self,
        path: Optional[str] = None,
        network_architecture_implementation_folder_path: Optional[str] = None,
        network_architecture_frame_folder_path: Optional[str] = None,
        load_from_dict: Optional[dict] = None,
    ):
        """
        Initialize the Architecture instance.
        
        Args:
            path: Path to the TOML configuration file
            network_architecture_implementation_folder_path: Path to network implementation folder
            network_architecture_frame_folder_path: Path to network frame folder
            load_from_dict: Optional dictionary to load configuration from
        """
        if load_from_dict is not None:
            object.__setattr__(self, "pop", POPArchitecture(**load_from_dict["pop"]))
            object.__setattr__(
                self, "agent", AgentArchitecture(load_from_dict=load_from_dict["agent"])
            )
            object.__setattr__(
                self,
                "manager",
                AgentArchitecture(load_from_dict=load_from_dict["manager"]),
            )
            object.__setattr__(
                self,
                "head_manager",
                AgentArchitecture(load_from_dict=load_from_dict["head_manager"]),
            )
            return
        architecture_dict: ParsedTOMLDict = toml.load(open(path))

        assert "pop" in architecture_dict.keys()
        assert "agent" in architecture_dict.keys()
        assert "manager" in architecture_dict.keys()
        assert "head_manager" in architecture_dict.keys()

        object.__setattr__(self, "pop", POPArchitecture(**architecture_dict["pop"]))
        object.__setattr__(
            self,
            "agent",
            AgentArchitecture(
                architecture_dict["agent"],
                network_architecture_implementation_folder_path=network_architecture_implementation_folder_path,
                network_architecture_frame_folder_path=network_architecture_frame_folder_path,
            ),
        )
        object.__setattr__(
            self,
            "manager",
            AgentArchitecture(
                architecture_dict["manager"],
                network_architecture_implementation_folder_path=network_architecture_implementation_folder_path,
                network_architecture_frame_folder_path=network_architecture_frame_folder_path,
            ),
        )
        object.__setattr__(
            self,
            "head_manager",
            AgentArchitecture(
                architecture_dict["head_manager"],
                network_architecture_implementation_folder_path=network_architecture_implementation_folder_path,
                network_architecture_frame_folder_path=network_architecture_frame_folder_path,
            ),
        )
