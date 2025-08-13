"""Configuration for training the model."""

from typing import Literal

from pydantic import BaseModel, Field

from baynext.config.priors import RoiMConfig
from baynext.config.sampling import SamplePosteriorConfig, SamplePriorConfig


class PriorConfig(BaseModel):
    """Configuration for prior distributions."""

    roi_m: RoiMConfig = RoiMConfig()


class ModelSpecConfig(BaseModel):
    """Configuration for the model specification."""

    prior: PriorConfig = PriorConfig()

    media_effects_dist: Literal["normal", "log_normal"] = "log_normal"
    hill_before_adstock: bool = False
    max_lag: int | None = Field(
        8,
        ge=0,
        description="Maximum number of lag periods (≥ 0) to include "
        "in the Adstock calculation",
    )


class TrainConfig(BaseModel):
    """Configuration for training the model."""

    spec: ModelSpecConfig = ModelSpecConfig()
    sample_prior: SamplePriorConfig = SamplePriorConfig()
    sample_posterior: SamplePosteriorConfig = SamplePosteriorConfig()
