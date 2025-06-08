import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import ray
from ray import ObjectRef
import torch as th

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture
from pop.networks.dueling_net import DuelingNet

import logging
import warnings

from pop.constants import PER_PROCESS_GPU_MEMORY_FRACTION

# Configure logging to suppress lightning framework logs           | Hung |
# This prevents interference with our own logging system          | Hung |
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

# Suppress user warnings to keep output clean                       | Hung |
warnings.filterwarnings("ignore", category=UserWarning)


"""
This module implements a distributed Graph Convolutional Network (GCN) agent using Ray framework.
Key features:
- Distributed training across multiple processes and GPUs
- Parallel processing of GCN-based reinforcement learning tasks
- State management for distributed environments
- Integration with Ray's distributed computing capabilities

The agent extends BaseGCNAgent to add distributed computing features while maintaining
the core GCN functionality for reinforcement learning tasks.
| Hung |
"""

# @ray.remote decorator marks this class for distributed execution      | Hung |
# num_cpus: CPU resources allocated per instance (0.5 means 2 instances per CPU)       | Hung |
# num_gpus: GPU resources allocated per instance (0 if no GPU, otherwise uses configured fraction)     | Hung |
@ray.remote(
    num_cpus=0.5,
    num_gpus=0 if not th.cuda.is_available() else PER_PROCESS_GPU_MEMORY_FRACTION,
)
class RayGCNAgent(BaseGCNAgent):
    """
    A distributed GCN agent that leverages Ray for parallel processing.
    Inherits from BaseGCNAgent and adds distributed computing capabilities.
    Manages distributed training and inference of GCN models across multiple 
    processes and GPUs.
    | Hung |
    """
    def __init__(
        self,
        agent_actions: int,  # Number of possible actions for the agent       | Hung |              
        node_features: int,  # Number of features per node in the graph       | Hung |
        architecture: AgentArchitecture,  # Network architecture configuration       | Hung |
        name: str,  # Unique identifier for the agent       | Hung |
        training: bool,  # Whether agent is in training mode       | Hung |
        device: str,  # Computing device (CPU/GPU)       | Hung |
        feature_ranges: Dict[str, Tuple[float, float]],  # Valid ranges for features       | Hung |
        edge_features: Optional[int] = None,  # Optional number of edge features       | Hung |
    ):
        """
        Initialize the distributed GCN agent.
        
        Sets up the base GCN agent with distributed computing capabilities
        and configures the network architecture for distributed training.
        | Hung |
        """
        BaseGCNAgent.__init__(
            self,
            agent_actions=agent_actions,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            training=training,
            device=device,
            tensorboard_dir=None,
            log_dir=None,
            feature_ranges=feature_ranges,
        )

    def get_q_network(self) -> DuelingNet:
        """
        Returns the Q-network used for action-value estimation.
        This network is used to predict the value of different actions.
        | Hung |
        """
        return self.q_network

    def get_name(self) -> str:
        """
        Returns the name identifier of the agent.
        Used for logging and identification in distributed environment.
        | Hung |
        """
        return self.name

    def reset_decay(self):
        """
        Resets the decay steps counter used for exploration rate decay.
        This helps in managing the exploration-exploitation trade-off
        during training.
        | Hung |
        """
        self.decay_steps = 0

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> ObjectRef:
        """
        Creates a new distributed GCN agent instance from a checkpoint.
        
        This factory method:
        - Creates a new agent instance with saved configuration
        - Loads network states and optimizer states
        - Restores training parameters
        - Returns a Ray object reference for distributed access
        
        The .remote() call creates a distributed instance of the agent
        that can be accessed across the Ray cluster.
        | Hung |
        """
        # Create a new distributed agent instance using Ray         | Hung |
        # The .remote() call makes this agent available across the Ray cluster   | Hung |
        agent: ObjectRef = RayGCNAgent.remote(
            agent_actions=checkpoint["agent_actions"],
            node_features=checkpoint["node_features"],
            architecture=AgentArchitecture(load_from_dict=checkpoint["architecture"])
            if kwargs.get("architecture") is None
            #architecture=AgentArchitecture(
            #    load_from_dict=checkpoint["architecture"]
            #) if kwargs.get("architecture") is None      | Hung |      
            else kwargs["architecture"],
            name=checkpoint["name"],
            training=bool(kwargs.get("training")),
            device=checkpoint["device"],
            edge_features=checkpoint["edge_features"],
            feature_ranges=checkpoint["feature_ranges"],
        )
        
        # Load the saved state into the distributed agent         | Hung |
        # The .remote() call makes this operation distributed       | Hung |
        agent.load_state.remote(
            optimizer_state=checkpoint["optimizer_state"],
            q_network_state=checkpoint["q_network_state"],
            target_network_state=checkpoint["target_network_state"],
            memory=checkpoint["memory"],
            exploration=checkpoint["exploration"],
            alive_steps=checkpoint["alive_steps"],
            train_steps=checkpoint["train_steps"],
            learning_steps=checkpoint["learning_steps"],
            reset_exploration=kwargs["reset_exploration"],
        )
        return agent
