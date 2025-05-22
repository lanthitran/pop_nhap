from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional

import toml

from pop.configs.placeholders_handling import (
    replace_backward_reference,
    replace_placeholders,
)
from pop.configs.type_aliases import ParsedTOMLDict
"""
This module defines the core data structures for representing neural network architectures in the POP framework.
It provides a flexible way to define and load network architectures from TOML configuration files.

The module consists of two main classes:
1. NetworkLayer: Represents a single layer in a neural network with its properties and parameters
2. NetworkArchitecture: Manages a collection of network layers and handles loading architectures
   from TOML files with support for placeholders and backward references

The architecture loading system supports two approaches:
- Direct loading from a dictionary of layer configurations
- Loading from TOML files with a frame-implementation pattern where:
  * Implementation files contain actual parameter values
  * Frame files define the structure with placeholders and references

This module is a crucial part of the POP framework as it:
- Provides a structured way to define neural network architectures
- Enables configuration-driven network design through TOML files
- Supports template-based architecture definitions with placeholders
- Allows for dynamic parameter resolution through backward references
"""

@dataclass(frozen=True)
class NetworkLayer:
    """
    Represents a single layer in a neural network architecture.
    
    Attributes:
        name (str): Unique identifier for the layer
        type (str): Type of the neural network layer (e.g., 'conv', 'linear')
        module (str): Python module path where the layer implementation can be found
        kwargs (Dict): Dictionary of keyword arguments for layer initialization
    """
    name: str
    type: str
    module: str
    kwargs: Dict[str, Union[int, float, bool, str]]


@dataclass(frozen=True)
class NetworkArchitecture:
    """
    Manages a collection of neural network layers and handles architecture loading.
    
    This class provides functionality to:
    - Load network architectures from TOML configuration files
    - Support placeholder replacement and backward references
    - Create a structured representation of the network architecture
    
    Attributes:
        layers (List[NetworkLayer]): List of network layers in the architecture
    """
    layers: List[NetworkLayer]

    def __init__(
        self,
        load_from_dict: Optional[Dict[str, List[ParsedTOMLDict]]] = None,
        network: Optional[str] = None,
        implementation_folder_path: Optional[str] = None,
        frame_folder_path: Optional[str] = None,
    ):
        """
        Initialize a NetworkArchitecture instance.
        
        Args:
            load_from_dict: Optional dictionary containing layer configurations
            network: Name of the network configuration to load
            implementation_folder_path: Path to folder containing implementation TOML files
            frame_folder_path: Path to folder containing frame TOML files
        
        The initialization supports two modes:
        1. Direct loading from a dictionary of layer configurations
        2. Loading from TOML files using a frame-implementation pattern
        """
        if load_from_dict:
            object.__setattr__(
                self,
                "layers",
                [NetworkLayer(**layer) for layer in load_from_dict["layers"]],
            )
            return

        if (
            network is None
            or implementation_folder_path is None
            or frame_folder_path is None
        ):
            raise Exception("Please pass either layers or all the other parameters")

        # Loading implementation value with actual architecture values
        network_architecture_implementation_dict: Dict[
            str, Union[str, Dict[str, Union[int, float, str, bool]]]
        ] = toml.load(open(Path(implementation_folder_path, network + ".toml")))

        assert "frame" in network_architecture_implementation_dict.keys()

        # Loading architecture frame with placeholders and back references
        # Placeholders are replaced with implementation values
        network_architecture_frame_dict: Dict[str, Dict[str, str]] = toml.load(
            open(
                Path(
                    frame_folder_path,
                    network_architecture_implementation_dict["frame"] + ".toml",
                )
            )
        )

        assert set(
            {k for k in network_architecture_implementation_dict.keys() if k != "frame"}
        ).issubset(network_architecture_frame_dict.keys())

        no_placeholder_architecture: Dict[
            str, Dict[str, int, bool, float]
        ] = replace_placeholders(
            implementation_dict=network_architecture_implementation_dict,
            frame_dict=network_architecture_frame_dict,
        )

        full_architecture = {
            layer_name: {
                layer_param_name: replace_backward_reference(
                    no_placeholder_architecture,
                    layer_param_value,
                    evaluate_expressions=True,
                )
                for layer_param_name, layer_param_value in layer_param_dict.items()
            }
            for layer_name, layer_param_dict in no_placeholder_architecture.items()
        }

        object.__setattr__(
            self,
            "layers",
            [
                NetworkLayer(
                    **{
                        "name": layer_name,
                        "type": layer_params["type"],
                        "module": layer_params["module"],
                        "kwargs": {
                            layer_param_name: layer_param_value
                            for layer_param_name, layer_param_value in layer_params.items()
                            if layer_param_name not in {"layer", "module"}
                        },
                    }
                )
                for layer_name, layer_params in full_architecture.items()
            ],
        )
