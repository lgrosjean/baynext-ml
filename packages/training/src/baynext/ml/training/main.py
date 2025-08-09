"""Module for Meridian model training.

This module provides functions to load input data, prepare the model specification,
train the model, and save the trained model.
"""

from .logger import get_logger
from .tasks import load, prepare, save, train

logger = get_logger(__name__)


def main(
    csv_path: str,
    kpi_type: str,
    time: str,
    kpi: str,
    controls: list[str],
    geo: str,
    population: str,
    roi_mu: float,
    roi_sigma: float,
    max_lag: int,
    n_draws: int,
    n_chains: int,
    n_adapt: int,
    n_burnin: int,
    n_keep: int,
    file_path: str,
    revenue_per_kpi: str | None = None,
    media: list[str] | None = None,
    media_spend: list[str] | None = None,
    organic_media: list[str] | None = None,
    non_media_treatments: list[str] | None = None,
) -> None:
    """Load, prepare, train and save the Meridian model with the specified parameters.

    Args:
        csv_path: Path to the CSV file containing input data.
        kpi_type: Type of KPI to analyze.
        time: Time column in the data.
        kpi: KPI column in the data.
        controls: List of control variables.
        geo: Geographic variable.
        population: Population variable.
        revenue_per_kpi: Revenue per KPI column, if applicable.
        media: List of media channels, if applicable.
        media_spend: List of media spend channels, if applicable.
        organic_media: List of organic media channels, if applicable.
        non_media_treatments: List of non-media treatments, if applicable.
        roi_mu: Mean of the ROI prior distribution.
        roi_sigma: Standard deviation of the ROI prior distribution.
        max_lag: Maximum lag for the model.
        n_draws: Number of draws for prior sampling.
        n_chains: Number of chains for posterior sampling.
        n_adapt: Number of adaptation steps for MCMC sampling.
        n_burnin: Number of burn-in steps for MCMC sampling.
        n_keep: Number of samples to keep after burn-in.
        file_path: Path to save the trained model.

    Returns:
        None: The trained model is saved to the specified file path.

    """
    logger.info("🚀 Starting Meridian model training pipeline...")

    input_data = load(
        csv_path=csv_path,
        kpi_type=kpi_type,
        time=time,
        kpi=kpi,
        controls=controls,
        geo=geo,
        population=population,
        revenue_per_kpi=revenue_per_kpi,
        media=media,
        media_spend=media_spend,
        organic_media=organic_media,
        non_media_treatments=non_media_treatments,
    )

    logger.info("✅ Input data loaded successfully.")

    model_spec = prepare(
        roi_mu=roi_mu,
        roi_sigma=roi_sigma,
        max_lag=max_lag,
    )

    logger.info("✅ Model specification prepared successfully.")

    meridian = train(
        input_data=input_data,
        model_spec=model_spec,
        n_draws=n_draws,
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep,
    )

    logger.info("✅ Meridian model trained successfully.")

    save(meridian, file_path)

    logger.info("✅ Trained model saved to %s", file_path)
