"""Module for training tasks.

This module provides functions to load data, prepare model specifications,
save the trained model, and train the Meridian model.
"""

from .load import task as load
from .prepare import task as prepare
from .save import task as save
from .train import task as train

__all__ = [
    "load",
    "prepare",
    "save",
    "train",
]
