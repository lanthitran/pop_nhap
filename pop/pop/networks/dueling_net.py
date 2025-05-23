from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch as th
import torch.nn as nn
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, FloatTensor

from pop.configs.network_architecture import NetworkArchitecture
from pop.networks.network_architecture_parsing import get_network
from pop.networks.serializable_module import SerializableModule, T
from pop.networks.gcn import GCN

"""
Dueling Network architecture for Deep Q-Learning with Graph Neural Networks.

This class implements a dueling network architecture that combines:
1. A graph convolutional network (GCN) for feature extraction
2. A value stream to estimate state values
3. An advantage stream to estimate action advantages

The network processes graph-structured data to output Q-values for each possible action.

Key components:
- GCN embedding layer for graph feature extraction
- Value stream for state value estimation
- Advantage stream for action advantage estimation
- Q-value computation combining value and advantage streams

Input: DGLHeteroGraph containing node and edge features
Output: Q-values for each action in the action space
| Hung |
"""

class DuelingNet(nn.Module, SerializableModule):
    """
    Dueling Network implementation combining GCN with value and advantage streams.
    
    Args:
        action_space_size: Number of possible actions
        node_features: Dimension of node features
        embedding_architecture: Architecture for GCN embedding layer
        advantage_stream_architecture: Architecture for advantage estimation
        value_stream_architecture: Architecture for value estimation
        feature_ranges: Valid ranges for input features
        name: Network identifier
        log_dir: Directory for logging
        edge_features: Optional dimension of edge features
    | Hung |
    """
    def __init__(
        self,
        action_space_size: int,
        node_features: int,
        embedding_architecture: NetworkArchitecture,
        advantage_stream_architecture: NetworkArchitecture,
        value_stream_architecture: NetworkArchitecture,
        feature_ranges: Dict[str, Tuple[float, float]],
        name: str,
        log_dir: Optional[str] = None,
        edge_features: Optional[int] = None,
    ):
        nn.Module.__init__(self)
        SerializableModule.__init__(self, log_dir, name)

        self.name = name

        # Parameters
        self.action_space_size = action_space_size

        # Embeddings
        self.embedding: GCN = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=embedding_architecture,
            name=name + "_embedding",
            log_dir=None,
            feature_ranges=feature_ranges,
        )

        # This attribute is used for reflection
        # Do not remove it
        self.embedding_size: int = self.embedding.get_embedding_dimension()

        self._node_embeddings: Optional[Tensor] = None

        self.advantage_stream: nn.Sequential = get_network(
            self, advantage_stream_architecture, is_graph_network=False
        )
        self.advantage_stream_architecture = advantage_stream_architecture

        self.value_stream: nn.Module = get_network(
            self, value_stream_architecture, is_graph_network=False
        )
        self.value_stream_architecture = value_stream_architecture

    def get_embedding_size(self) -> int:
        """
        Returns the dimension of the graph embeddings.
        
        Returns:
            int: Dimension of the embedding space
        | Hung |
        """
        return self.embedding.get_embedding_dimension()

    @staticmethod
    def _compute_graph_embedding(
        g: DGLHeteroGraph, node_embeddings: th.Tensor
    ) -> th.Tensor:
        """
        Computes graph-level embeddings by averaging node embeddings.
        
        Args:
            g: Input graph
            node_embeddings: Node-level embeddings
            
        Returns:
            Tensor: Graph-level embedding
        | Hung |
        """
        # -> (node*batch_size, embedding_size)
        g.ndata["node_embeddings"] = node_embeddings

        # -> (batch_size, embedding_size)
        graph_embedding: Tensor = dgl.mean_nodes(g, "node_embeddings")
        del g.ndata["node_embeddings"]
        return graph_embedding

    def _extract_features(self, g: DGLHeteroGraph) -> Tensor:
        """
        Extracts features from input graph using GCN and computes graph embedding.
        
        Args:
            g: Input graph
            
        Returns:
            Tensor: Flattened graph embedding
        | Hung |
        """
        # (node*batch_size, embedding_size)
        self._node_embeddings = self.embedding(g)

        # -> (batch_size, embedding_size)
        graph_embedding = self._compute_graph_embedding(g, self._node_embeddings)

        graph_embedding = th.flatten(graph_embedding, 1)

        # -> (batch_size, embedding_size)
        return graph_embedding

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        """
        Forward pass computing Q-values for each action.
        
        Args:
            g: Input graph
            
        Returns:
            Tensor: Q-values for each action
        | Hung |
        """
        # -> (batch_size, embedding_size)
        graph_embedding: Tensor = self._extract_features(g)

        # Compute value of current state
        # -> (batch_size, 1)
        state_value: float = self.value_stream(graph_embedding)

        # Compute advantage of (current_state, action) for each action
        # -> (batch_size, action_space_size)
        state_advantages: FloatTensor = self.advantage_stream(graph_embedding)

        # -> (batch_size, action_space_size)
        q_values: Tensor = state_value + (state_advantages - state_advantages.mean())

        # -> (batch_size, action_space_size)
        return q_values

    def advantage(self, g: DGLHeteroGraph) -> Tensor:
        """
        Computes advantage values for each action.

        Args:
            g: Input graph
            
        Returns:
            Tensor: Advantage values averaged across batch
        | Hung |
        """
        # -> (batch_size, embedding_size)
        graph_embedding: Tensor = self._extract_features(g)

        # Compute advantage of (current_state, action) for each action
        # -> (batch_size, action_space_size)
        state_advantages: Tensor = self.advantage_stream(graph_embedding)

        # -> (action_space_size)
        state_advantages: Tensor = th.mean(state_advantages, 0)

        # -> (action_space_size)
        return state_advantages

    def get_state(self: T) -> Dict[str, Any]:
        """
        Returns the current state of the network for serialization.
        
        Returns:
            Dict containing network state and architecture
        | Hung |
        """
        return {
            "name": self.name,
            "action_space_size": self.action_space_size,
            "embedding_state": self.embedding.get_state(),
            "advantage_stream_architecture": asdict(self.advantage_stream_architecture),
            "advantage_stream_state": self.advantage_stream.state_dict(),
            "value_stream_architecture": asdict(self.value_stream_architecture),
            "value_stream_state": self.value_stream.state_dict(),
            "log_file": self.log_file,
        }

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> "DuelingNet":
        """
        Creates a new DuelingNet instance from a checkpoint.
        
        Args:
            checkpoint: Dictionary containing saved network state
            
        Returns:
            DuelingNet: Reconstructed network
        | Hung |
        """
        dueling_net: "DuelingNet" = DuelingNet(
            action_space_size=checkpoint["action_space_size"],
            node_features=checkpoint["embedding_state"]["node_features"],
            edge_features=checkpoint["embedding_state"]["edge_features"],
            embedding_architecture=NetworkArchitecture(
                load_from_dict=checkpoint["embedding_state"]["architecture"]
            ),
            advantage_stream_architecture=NetworkArchitecture(
                load_from_dict=checkpoint["advantage_stream_architecture"]
            ),
            value_stream_architecture=NetworkArchitecture(
                load_from_dict=checkpoint["value_stream_architecture"]
            ),
            log_dir=str(Path(checkpoint["log_file"]).parents[0])
            if checkpoint["log_file"] is not None
            else None,
            name=checkpoint["name"],
        )
        dueling_net.embedding.load_state_dict(
            checkpoint["embedding_state"]["network_state"]
        )
        dueling_net.advantage_stream.load_state_dict(
            checkpoint["advantage_stream_state"]
        )
        dueling_net.value_stream.load_state_dict(checkpoint["value_stream_state"])
        return dueling_net
