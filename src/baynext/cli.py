"""Defines CLI Baynext training."""

from baynext.config.pipeline import PipelineConfig
from baynext.pipeline import pipeline


def cli() -> None:
    """Baynext CLI entrypoint."""
    pipeline_config = PipelineConfig()

    pipeline(pipeline_config)
