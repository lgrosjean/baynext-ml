"""Logging utility for the training module."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger("training")
    # Each call to get_logger adds a new handler,
    # which can lead to duplicate log entries.
    # Consider checking for existing handlers
    # before adding or configuring the logger once at module import.
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger.getChild(name)
