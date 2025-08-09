"""Module for Meridian model training.

This module provides functions to prepare loaded data for the model specification.
"""

import tensorflow_probability as tfp
from meridian.model import prior_distribution, spec

from training.logger import get_logger
from utils.constants import ROI_M

logger = get_logger(__name__)


def task(roi_mu: float, roi_sigma: float, max_lag: int) -> spec.ModelSpec:
    """Prepare the model specification with prior distribution.

    Args:
        roi_mu: Mean of the ROI prior distribution
        roi_sigma: Standard deviation of the ROI prior distribution
        max_lag: Maximum lag for the model

    Returns:
        ModelSpec: The model specification with the prior distribution set

    """
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=ROI_M),
    )
    return spec.ModelSpec(prior=prior, max_lag=max_lag)
