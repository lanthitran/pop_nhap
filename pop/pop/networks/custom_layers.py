from typing import Tuple

import torch.nn as nn
import torch as th
from dgl.nn.pytorch import GraphConv
from torch import Tensor

from dgl import DGLHeteroGraph


"""
This module implements custom neural network layers for graph attention networks (GAT) 
and edge-augmented graph attention networks (EGAT).

Key components:
- EGATFlatten: Flattens both node and edge embeddings
- GATFlatten: Flattens only node embeddings 
- EGATNodeConv: Performs graph convolution with edge weights

These layers are essential for processing graph-structured data in the power network
optimization framework.
| Hung |
"""

class EGATFlatten(nn.Module):
    """
    Flattens both node and edge embeddings in an edge-augmented graph attention network.
    
    This layer takes node and edge embeddings and flattens them along the feature 
    dimension to prepare them for downstream processing.
    
    Input shapes:
    - node_embedding: (nodes, node_features, heads, batch_size)
    - edge_embedding: (edges, edge_features, heads, batch_size)
    
    Output shapes:
    - node_embedding: (nodes, node_features * heads, batch_size)
    - edge_embedding: (edges, edge_features * heads, batch_size)
    | Hung |
    """
    def __init__(self):
        super(EGATFlatten, self).__init__()

    def forward(
        self, g: DGLHeteroGraph, node_embedding: Tensor, edge_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # -> (nodes, node_features * heads, batch_size),
        # -> (edges, edge_features * heads, batch_size)
        return th.flatten(node_embedding, 1), th.flatten(edge_embedding, 1)


class GATFlatten(nn.Module):
    """
    Flattens node embeddings in a graph attention network.
    
    This layer takes only node embeddings and flattens them along the feature dimension
    to prepare them for downstream processing.
    
    Input shape:
    - node_embedding: (nodes, node_features, heads, batch_size)
    
    Output shape:
    - node_embedding: (nodes, node_features * heads, batch_size)
    | Hung |
    """
    def __init__(self):
        super(GATFlatten, self).__init__()

    def forward(self, g: DGLHeteroGraph, node_embedding: Tensor):
        # -> (nodes, node_features * heads, batch_size)
        return th.flatten(node_embedding, 1)


class EGATNodeConv(nn.Module):
    """
    Performs graph convolution with edge weights in an edge-augmented graph attention 
    network.
    
    This layer applies graph convolution while incorporating edge features as weights
    to capture the influence of edges on node representations.
    
    Args:
        in_feats: Number of input features
        out_feats: Number of output features
        bias: Whether to use bias term
        allow_zero_in_degree: Whether to allow nodes with zero in-degree
    
    Input shapes:
    - node_embedding: (nodes, node_features, batch_size)
    - edge_embedding: (edges, edge_features, batch_size)
    
    Output shape:
    - node_embedding: (nodes, node_features, batch_size)
    | Hung |
    """
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        bias=True,
        allow_zero_in_degree=True,
    ):
        super(EGATNodeConv, self).__init__()
        self.convolution: GraphConv = GraphConv(
            in_feats,
            out_feats,
            bias=bias,
            allow_zero_in_degree=allow_zero_in_degree,
        )

    def forward(
        self, g: DGLHeteroGraph, node_embedding: Tensor, edge_embedding: Tensor
    ) -> Tensor:
        # -> (nodes, node_features, batch_size)
        return self.convolution(g, node_embedding, edge_weight=edge_embedding)
