from typing import Union

from pop.configs.network_architecture import NetworkLayer, NetworkArchitecture
import re


from typing import TYPE_CHECKING
import numexpr as ne

if TYPE_CHECKING:
    from pop.networks.gcn import GCN
    from networks.dueling_net import DuelingNet

# Necessary imports (with aliases) for parsing
import torch.nn as nn
import dgl.nn.pytorch as dgl_nn
import pop.networks.custom_layers as cl  # pylint: disable=unused-import

"""
This module provides functions for dynamically constructing neural network architectures based on configuration.
It is used to parse network architecture specifications and create corresponding PyTorch/DGL models.

Key functions:
- get_network: Creates a sequential model (either DGL or PyTorch) from an architecture specification
- _parse_layer: Converts a NetworkLayer configuration into executable Python code for layer creation
- _replace_dynamic_placeholder: Replaces dynamic placeholders in layer configurations with actual values

The module supports:
- Graph neural networks (DGL)
- Standard PyTorch networks
- Custom network layers
- Dynamic parameter resolution
- Mathematical expressions in configurations
"""

def get_network(
    self: Union["GCN", "DuelingNet", "InverseNetwork"],
    architecture: NetworkArchitecture,
    is_graph_network: bool,
) -> dgl_nn.Sequential:
    """
    Creates a sequential neural network model based on the provided architecture specification.
    
    Args:
        self: The network instance (GCN, DuelingNet, or InverseNetwork)
        architecture: NetworkArchitecture object containing layer configurations
        is_graph_network: Whether to create a DGL graph network or standard PyTorch network
        
    Returns:
        A sequential model containing the configured layers
    """
    if is_graph_network:
        model: dgl_nn.Sequential = dgl_nn.Sequential()
    else:
        model: nn.Sequential = nn.Sequential()
    for layer in architecture.layers:
        model.append(eval(_parse_layer(self, layer)))

    return model


def _parse_layer(self: Union["GCN", "DuelingNet"], layer: NetworkLayer) -> str:
    """
    Converts a NetworkLayer configuration into executable Python code for layer creation.
    
    Args:
        self: The network instance
        layer: NetworkLayer configuration specifying layer type and parameters
        
    Returns:
        String representation of layer initialization code
    """
    return (
        (
            "dgl_nn"
            if layer.module == "dgl"
            else "nn"
            if layer.module == "pytorch"
            else "cl"
        )
        + "."
        + layer.type
        + "("
        + ",".join(
            [
                arg_name
                + "="
                + (
                    (_replace_dynamic_placeholder(self, str(arg_value)))
                    if arg_name != "activation"
                    else ("nn." + str(arg_value) + "()")
                )
                for arg_name, arg_value in layer.kwargs.items()
                if arg_name not in {"type", "module"}
            ]
        )
        + ")"
    )


def _replace_dynamic_placeholder(
    self: Union["GCN", "DuelingNet"], placeholder: str
) -> str:
    """
    Replaces dynamic placeholders in layer configurations with actual values.
    Supports mathematical expressions that are evaluated using numexpr.
    
    Args:
        self: The network instance
        placeholder: String containing placeholders in format <attribute_name>
        
    Returns:
        String with placeholders replaced by actual values
    """
    while re.match(r"<\w*>", placeholder):
        open_bracket = placeholder.find("<")
        closed_bracket = placeholder.find(">")
        attribute = placeholder[open_bracket + 1 : closed_bracket]
        attribute_value = self.__getattribute__(attribute)
        placeholder = placeholder.replace(
            placeholder[open_bracket : closed_bracket + 1], str(attribute_value)
        )

    return str(ne.evaluate(placeholder))
