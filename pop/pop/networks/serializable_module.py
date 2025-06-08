import sys
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, TypeVar
import torch as th
from pathlib import Path

"""
This module provides a base class for serializing and deserializing PyTorch models.
It's a crucial component for saving training progress and loading pre-trained models.

Key concepts:
- Checkpoint: A saved state of the model that can be loaded later
- Log file: The base path where checkpoints are stored
- Counter: A number appended to checkpoint files to track versions

The module uses PyTorch's save/load functionality and integrates with the project's
logging system to manage model persistence.

This module is essential for:
1. Saving model states during training
2. Loading pre-trained models
3. Resuming training from checkpoints
4. Managing model versioning
| Hung |
"""

T = TypeVar("T")  # Generic type variable for type hints | Hung |


class SerializableModule(ABC):
    """
    SerializableModule is an abstract base class that provides functionality for saving and loading model checkpoints.
    It is used throughout the project to enable model persistence and resumable training.

    Key features:
    - Manages checkpoint file paths and naming
    - Handles incremental checkpoint saving with counter suffixes
    - Provides abstract methods for state serialization/deserialization
    - Implements robust checkpoint loading with fallback to previous versions
    - Used by BasePOP and other classes that need checkpoint functionality

    The class is designed to work with PyTorch's save/load functionality and integrates
    with the project's logging system.

    Usage:
    1. Inherit from this class
    2. Implement get_state() and factory() methods
    3. Use save() to save checkpoints
    4. Use load() to load checkpoints
    | Hung |
    """
    def __init__(self, log_dir: Optional[str], name: Optional[str]):
        """
        Initialize the SerializableModule with logging directory and name.
        
        Args:
            log_dir: Directory where checkpoints will be saved
            name: Name of the model/checkpoint file
        | Hung |
        """
        self.log_file = self._get_log_file(log_dir, name)
        self.number_of_saves = 0

    @staticmethod
    def _get_log_file(
        log_dir: Optional[str], file_name: Optional[str]
    ) -> Optional[str]:
        """
        Generate the log file path from directory and name.
        
        Args:
            log_dir: Directory for saving checkpoints
            file_name: Name of the checkpoint file
            
        Returns:
            Full path to the checkpoint file or None if log_dir is None
        | Hung |
        """
        if log_dir is not None:
            if file_name is None:
                raise Exception("Please pass a non-null name to get_log_file")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            return str(Path(log_dir, file_name + ".pt"))
        return None

    @abstractmethod
    def get_state(self: T) -> Dict[str, Any]:
        """
        Abstract method that must be implemented to define what state to save.
        
        Returns:
            Dictionary containing the state to be saved
        | Hung |
        """
        ...

    @staticmethod
    def _add_counter_to_file_path(log_file: str, counter: int) -> str:
        """
        Add a counter suffix to the checkpoint filename.
        
        Args:
            log_file: Base checkpoint file path
            counter: Counter value to append
            
        Returns:
            New file path with counter suffix
        | Hung |
        """
        log_file_path = Path(log_file)
        log_file_path_name_split = log_file_path.name.split(".")
        return str(
            Path(
                log_file_path.parents[0],
                log_file_path_name_split[0]
                + "_"
                + str(counter)
                + "."
                + ".".join(log_file_path_name_split[1:]),
            )
        )

    @staticmethod
    def _get_last_saved_checkpoint(log_file: str) -> int:
        """
        Find the highest checkpoint number in the directory.
        
        Args:
            log_file: Base checkpoint file path
            
        Returns:
            Highest checkpoint number found
        | Hung |
        """
        return max(
            [
                int(dir_object.stem.split("_")[-1])
                for dir_object in Path(log_file).parents[0].iterdir()
                if dir_object.is_file()
            ]
        )

    def save(self: T) -> None:
        """
        Save the current state as a checkpoint.
        
        The checkpoint is saved with an incremental counter suffix.
        If this is the first save and previous checkpoints exist,
        the counter will continue from the last checkpoint.
        | Hung |
        """
        if self.log_file is None:
            raise Exception("Called save() in " + self.name + " with None log_dir")

        checkpoint = self.get_state()
        if self.number_of_saves == 0 and list(Path(self.log_file).parents[0].iterdir()):
            self.number_of_saves = self._get_last_saved_checkpoint(self.log_file) + 2
        else:
            self.number_of_saves += 1

        th.save(
            checkpoint,
            self._add_counter_to_file_path(self.log_file, self.number_of_saves - 1),
        )

    @classmethod
    def load(
        cls,
        log_file: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> T:
        """
        Load a checkpoint and create a new instance.
        
        Args:
            log_file: Path to the checkpoint file
            checkpoint: Pre-loaded checkpoint dictionary
            **kwargs: Additional arguments for factory method
            
        Returns:
            New instance with loaded state
        | Hung |
        """
        checkpoint: Dict[str, Any] = SerializableModule._load_checkpoint(
            log_file, checkpoint
        )
        return cls.factory(checkpoint, **kwargs)

    @staticmethod
    @abstractmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> T:
        """
        Abstract method that must be implemented to create a new instance from checkpoint.
        
        Args:
            checkpoint: Dictionary containing saved state
            **kwargs: Additional arguments for initialization
            
        Returns:
            New instance with loaded state
        | Hung |
        """
        ...

    @staticmethod
    def _load_checkpoint(
        log_file: Optional[str], checkpoint_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Load checkpoint data from file or dictionary.
        
        If loading from file fails, it will try previous checkpoints.
        If no valid checkpoint is found, the program will exit.
        
        Args:
            log_file: Path to checkpoint file
            checkpoint_dict: Pre-loaded checkpoint dictionary
            
        Returns:
            Loaded checkpoint dictionary
        | Hung |
        """
        if log_file is None and checkpoint_dict is None:
            raise Exception(
                "Cannot load module: both log_file and checkpoint_dict are None"
            )
        if log_file is not None:
            last_saved_checkpoint = SerializableModule._get_last_saved_checkpoint(
                log_file
            )
            checkpoint_to_load = SerializableModule._add_counter_to_file_path(
                log_file, last_saved_checkpoint
            )
            print("Loaded Last Checkpoint: " + str(checkpoint_to_load))
            while last_saved_checkpoint >= 0:
                try:
                    return th.load(checkpoint_to_load)
                except Exception as e:
                    print(
                        "Exception encountered when loading checkpoint "
                        + str(last_saved_checkpoint)
                    )
                    print(e)

                last_saved_checkpoint -= 1
                checkpoint_to_load = SerializableModule._add_counter_to_file_path(
                    log_file, last_saved_checkpoint
                )

            print("There is no valid checkpoint left to reload")
            sys.exit(
                0
            )  # We usually run in an until loop in a bash script, this allows to break it

        return checkpoint_dict
