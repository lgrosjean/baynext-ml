"""Model spec priors config."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from baynext.config.distributions import Distribution, LogNormalDistributionConfig

if TYPE_CHECKING:
    from tensorflow_probability import distributions as tfp


class RoiMConfig(BaseModel):
    """Configuration for the ROI M distribution."""

    distribution: Distribution = LogNormalDistributionConfig()

    def to_tfp(self) -> "tfp.LogNormal":
        """Convert to TensorFlow Probability distribution."""
        return self.distribution.to_tfp(name="roi_m")
