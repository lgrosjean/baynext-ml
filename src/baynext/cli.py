"""Defines CLI Baynext training."""

from baynext.config.pipeline import PipelineConfig
from baynext.pipeline import run_pipeline


def cli() -> None:
    """Baynext CLI entrypoint."""
    pipeline_config = PipelineConfig()
    run_pipeline(pipeline_config)
