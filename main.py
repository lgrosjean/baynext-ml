"""Main entry point for the Baynext ML application."""

import logging
import os
import ssl

# Set TensorFlow logging level to minimize verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Suppress XLA logs
os.environ["XLA_FLAGS"] = "--xla_hlo_profile=false"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# Suppress JAX logs
os.environ["JAX_LOG_COMPILES"] = "0"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)

# Set specific loggers to WARNING level to suppress INFO logs
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
logging.getLogger("jax._src.compiler").setLevel(logging.WARNING)


import mlflow
from meridian.data import load
from meridian.mlflow import autolog
from meridian.model import model

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def pipeline() -> None:
    """Run the Meridian ML Training pipeline."""
    # Enable autologging (call this once per session)
    autolog.autolog(log_metrics=True)

    # Start an MLflow run (optionally name it for better grouping)
    with mlflow.start_run(run_name="my_run"):
        logger.info("⚡️ Starting Meridian ML pipeline...")

        dataset_source = "https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_media.csv"

        # Load data
        coord_to_columns = load.CoordToColumns(
            time="time",
            geo="geo",
            population="population",
            controls=["sentiment_score_control", "competitor_activity_score_control"],
            kpi="conversions",
            revenue_per_kpi="revenue_per_conversion",
            media=[
                "Channel0_impression",
                "Channel1_impression",
                "Channel2_impression",
                "Channel3_impression",
            ],
            media_spend=[
                "Channel0_spend",
                "Channel1_spend",
                "Channel2_spend",
                "Channel3_spend",
            ],
        )

        correct_media_to_channel = {
            "Channel0_impression": "Channel0",
            "Channel1_impression": "Channel1",
            "Channel2_impression": "Channel2",
            "Channel3_impression": "Channel3",
        }

        correct_media_spend_to_channel = {
            "Channel0_spend": "Channel0",
            "Channel1_spend": "Channel1",
            "Channel2_spend": "Channel2",
            "Channel3_spend": "Channel3",
        }

        loader = load.CsvDataLoader(
            csv_path=dataset_source,
            kpi_type="non_revenue",
            coord_to_columns=coord_to_columns,
            media_to_channel=correct_media_to_channel,
            media_spend_to_channel=correct_media_spend_to_channel,
        )

        data = loader.load()

        dataset = mlflow.data.from_pandas(
            loader._df_loader.df,
            source=dataset_source,
            name="national_media",
        )

        mlflow.log_input(dataset)

        logger.info("✅ Dataset loaded!")

        # Initialize Meridian model
        logger.info("🔧 Initializing Meridian model...")
        mmm = model.Meridian(input_data=data)

        n_draws = 100
        n_chains = 7
        n_adapt = 500
        n_burnin = 500
        n_keep = 1000

        # Run Meridian sampling processes
        logger.info("🔍 Running Meridian sampling processes...")
        mmm.sample_prior(n_draws=n_draws, seed=123)
        mmm.sample_posterior(
            n_chains=n_chains,
            n_adapt=n_adapt,
            n_burnin=n_burnin,
            n_keep=n_keep,
            seed=1,
        )

        logger.info("✅ Meridian sampling processes completed!")


if __name__ == "__main__":
    pipeline()
