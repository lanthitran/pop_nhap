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

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

warnings.filterwarnings("ignore", category=UserWarning)
"""
RayGCNAgent is a distributed agent class that extends BaseGCNAgent to provide 
distributed computing capabilities using Ray framework. It enables parallel 
processing of GCN-based reinforcement learning tasks across multiple processes 
and GPUs. The agent handles distributed training, inference, and state management 
for GCN models in a distributed environment.
| Hung |
"""

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
        agent_actions: int,
        node_features: int,
        architecture: AgentArchitecture,
        name: str,
        training: bool,
        device: str,
        feature_ranges: Dict[str, Tuple[float, float]],
        edge_features: Optional[int] = None,
    ):
        """
        Initialize the distributed GCN agent with network architecture and 
        training configuration. Sets up the base GCN agent with distributed 
        computing capabilities.
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
        | Hung |
        """
        return self.q_network

    def get_name(self) -> str:
        """
        Returns the name identifier of the agent.
        | Hung |
        """
        return self.name

    def reset_decay(self):
        """
        Resets the decay steps counter used for exploration rate decay.
        | Hung |
        """
        self.decay_steps = 0

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> ObjectRef:
        """
        Creates a new distributed GCN agent instance from a checkpoint.
        Handles loading of network states, optimizer states, and training 
        parameters. Returns a Ray object reference to the created agent.
        | Hung |
        """
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
