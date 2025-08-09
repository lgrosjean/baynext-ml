"""Train the Meridian model with specified parameters.

This module provides functions to train the Meridian model using input data
and model specifications.
"""

from meridian.data.input_data import InputData
from meridian.model.model import Meridian
from meridian.model.spec import ModelSpec

from training.logger import get_logger

logger = get_logger(__name__)


def task(
    input_data: InputData,
    model_spec: ModelSpec,
    n_draws: int,
    n_chains: int,
    n_adapt: int,
    n_burnin: int,
    n_keep: int,
):
    """Run the Meridian model with the given parameters.

    Source: https://developers.google.com/meridian/docs/user-guide/run-model
    """
    meridian = Meridian(input_data=input_data, model_spec=model_spec)
    meridian.sample_prior(n_draws=n_draws)
    meridian.sample_posterior(
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep,
    )
    return meridian
