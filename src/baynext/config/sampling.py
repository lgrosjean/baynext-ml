"""Configuration for sampling."""

from pydantic import BaseModel, Field

_N_DRAWS = 100
_N_CHAINS = 7
_N_ADAPT = 500
_N_BURNIN = 500
_N_KEEP = 1000


class SamplePriorConfig(BaseModel):
    """Configuration for prior sampling."""

    n_draws: int = Field(
        default=_N_DRAWS,
        description="Number of draws for sampling.",
        gt=0,
    )


class SamplePosteriorConfig(BaseModel):
    """Configuration for posterior sampling."""

    n_chains: int = _N_CHAINS
    n_adapt: int = _N_ADAPT
    n_burnin: int = _N_BURNIN
    n_keep: int = _N_KEEP
    max_tree_depth: int = Field(
        default=10,
        ge=0,
        description="Maximum tree depth for the model.",
    )
