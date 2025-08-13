"""Train Meridian model task."""

from meridian.data.input_data import InputData
from meridian.model import model, prior_distribution, spec

from baynext.config.pipeline import TrainConfig


def train_task(input_data: InputData, train_config: TrainConfig) -> model.Meridian:
    """Train the model based on the provided configuration."""
    prior = prior_distribution.PriorDistribution(
        roi_m=train_config.spec.prior.roi_m.to_tfp(),
    )
    model_spec = spec.ModelSpec(
        prior=prior,
        media_effects_dist=train_config.spec.media_effects_dist,
        hill_before_adstock=train_config.spec.hill_before_adstock,
    )
    mmm = model.Meridian(
        input_data=input_data,
        model_spec=model_spec,
    )

    mmm.sample_prior(
        **train_config.sample_prior.model_dump(),
        seed=123,
    )
    mmm.sample_posterior(
        **train_config.sample_posterior.model_dump(),
        seed=1,
    )

    return mmm
