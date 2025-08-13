"""Tasks modules."""

from baynext.tasks.load import load_task as load
from baynext.tasks.train import train_task as train

__all__ = [
    "load",
    "train",
]
