"""Save the Meridian model.

This module provides functions to save the trained Meridian model to a file.
"""

from meridian.model.model import Meridian, save_mmm

from training.logger import get_logger

logger = get_logger(__name__)


def task(meridian: Meridian, file_path: str) -> None:
    """Save the Meridian model to a file."""
    save_mmm(meridian, file_path)
