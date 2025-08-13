"""Logging configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogConfig(BaseModel):
    """Configuration for logging."""

    level: LogLevel = "INFO"
    metrics: Annotated[
        bool,
        Field(description="Enable training metrics logging"),
    ] = True
    """Enable logging of training metrics."""
    dataset: Annotated[
        bool,
        Field(description="Enable dataset logging"),
    ] = True
    """Enable logging of dataset."""
    model: Annotated[
        bool,
        Field(description="Enable model logging"),
    ] = True
    """Enable logging of model."""
    system_metrics: Annotated[
        bool,
        Field(description="Enable system metrics logging"),
    ] = True
    """Enable logging of system metrics."""
