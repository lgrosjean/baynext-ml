"""Defines distributions for Bayesian modeling."""

from abc import ABC, abstractmethod
from typing import Literal, Union

from pydantic import BaseModel, Field
from tensorflow_probability import distributions as tfp

_ROI_MU = 0.2
_ROI_SIGMA = 0.9


class DistributionConfig(BaseModel, ABC):
    """Base configuration for distributions."""

    type: Literal["log_normal", "normal"] = Field(
        description="Type of the distribution.",
    )
    """The type of the distribution."""

    @abstractmethod
    def to_tfp(self, name: str) -> tfp.Distribution:
        """Convert to TensorFlow Probability distribution."""


class LogNormalDistributionConfig(DistributionConfig):
    """Configuration for log-normal distributions."""

    type: Literal["log_normal"] = "log_normal"
    mu: float = Field(
        default=_ROI_MU,
        description="Mean of the log-normal distribution.",
    )
    """Means of the underlying Normal distribution(s)."""
    sigma: float = Field(
        default=_ROI_SIGMA,
        description="Standard deviation of the log-normal distribution.",
    )
    """ Stddevs of the underlying Normal distribution(s)."""

    def to_tfp(self, name: str) -> tfp.LogNormal:  # noqa: D102
        return tfp.LogNormal(loc=self.mu, scale=self.sigma, name=name)


class NormalDistributionConfig(DistributionConfig):
    """Configuration for normal distributions."""

    type: Literal["normal"] = "normal"
    mu: float = Field(
        default=_ROI_MU,
        description="Mean of the normal distribution.",
    )
    sigma: float = Field(
        default=_ROI_SIGMA,
        description="Standard deviation of the normal distribution.",
    )

    def to_tfp(self, name: str) -> tfp.Normal:  # noqa: D102
        return tfp.Normal(loc=self.mu, scale=self.sigma, name=name)


Distribution = LogNormalDistributionConfig | NormalDistributionConfig
