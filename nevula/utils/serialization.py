import os
import pickle
from typing import Any, Optional, Union
from nevula.nn.module import Module
from nevula.optim.optimizer import Optimizer


def save(obj: Any, f: Union[str, os.PathLike]) -> None:
    """
    Saves an object to a disk file using pickle serialization.

    Args:
        obj: The object to save (e.g. state_dict or model).
        f: File path to write to.
    """
    filepath = str(f)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load(f: Union[str, os.PathLike]) -> Any:
    """
    Loads an object saved with `save` from a file.

    Args:
        f: File path to read from.
    """
    filepath = str(f)
    with open(filepath, "rb") as fp:
        return pickle.load(fp)


def save_checkpoint(
    filepath: Union[str, os.PathLike],
    model: Union[Module, dict[str, Any]],
    optimizer: Optional[Union[Optimizer, dict[str, Any]]] = None,
    epoch: Optional[int] = None,
    **kwargs: Any
) -> None:
    """
    Saves a training checkpoint including model weights, optimizer state,
    and metadata.

    Args:
        filepath: Destination file path.
        model: Model instance or model state_dict.
        optimizer: Optimizer instance or optimizer state_dict (optional).
        epoch: Current epoch or step number (optional).
        **kwargs: Additional metadata to save in checkpoint.
    """
    model_state = model.state_dict() if hasattr(model, "state_dict") else model
    optimizer_state = optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else optimizer

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "epoch": epoch,
        **kwargs
    }
    save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, os.PathLike],
    model: Optional[Module] = None,
    optimizer: Optional[Optimizer] = None
) -> dict[str, Any]:
    """
    Loads a checkpoint and optionally restores model and optimizer states.

    Args:
        filepath: Path to checkpoint file.
        model: Model to load weights into (optional).
        optimizer: Optimizer to load state into (optional).

    Returns:
        The loaded checkpoint dictionary.
    """
    checkpoint = load(filepath)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        if checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
