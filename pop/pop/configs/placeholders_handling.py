import re
from typing import Dict, Union, List
import numexpr as ne


"""
This module handles placeholder and backward reference replacements in configuration dictionaries.
It provides functionality to resolve references between different layers and parameters in a neural network configuration.
| Hung |
"""

def replace_backward_reference(
    reference_dict: Dict[str, Dict[str, Union[int, bool, float, str]]],
    value_to_replace: Union[int, float, bool, str],
    evaluate_expressions: bool,
) -> Union[int, float, bool, str]:
    """
    Replaces backward references in a value with their corresponding values from the reference dictionary.
    
    Args:
        reference_dict: Dictionary containing layer parameters and their values
        value_to_replace: Value that may contain backward references to be replaced
        evaluate_expressions: Whether to evaluate mathematical expressions after replacement
        
    Returns:
        The value with all backward references replaced and optionally evaluated
    | Hung |
    """
    if not isinstance(value_to_replace, str):
        return value_to_replace

    # Backward references are "<<...>>" where ...="layerName_paramName"
    # A general param_value may contain multiple backward references arranged
    # as a mathematical expression
    backward_references: List[str] = re.findall(r"<<\w*>>", value_to_replace)

    if not backward_references:
        return value_to_replace

    for backward_reference in backward_references:

        # strip "<" and ">" from the references
        backward_reference_stripped: str = re.sub(r"[<>]", "", backward_reference)

        # split over "_"
        # First part is layer name
        # All the rest is layer parameter name
        split_backward_reference: List[str] = backward_reference_stripped.split("_")

        value_to_replace = value_to_replace.replace(
            backward_reference,
            str(
                reference_dict[split_backward_reference[0]][
                    "_".join(split_backward_reference[1:])
                ]
            ),
        )
    if evaluate_expressions:
        return ne.evaluate(value_to_replace)
    else:
        return value_to_replace


def replace_placeholders(
    implementation_dict: Dict[
        str,
        Union[str, Dict[str, Union[int, float, str, bool]]],
    ],
    frame_dict: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Union[int, bool, float, str]]]:
    """
    Replaces placeholder values ("...") in the frame dictionary with values from the implementation dictionary.
    
    Args:
        implementation_dict: Dictionary containing default implementation values
        frame_dict: Dictionary containing layer configurations with potential placeholders
        
    Returns:
        Dictionary with all placeholders replaced with their corresponding implementation values
    | Hung |
    """
    return {
        layer_name: {
            layer_param_name: (
                layer_param_value
                if not layer_param_value == "..."
                else implementation_dict[layer_name][layer_param_name]
            )
            for layer_param_name, layer_param_value in layer_param_dict.items()
        }
        for layer_name, layer_param_dict in frame_dict.items()
    }
