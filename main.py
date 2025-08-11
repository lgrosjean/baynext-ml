"""Main entry point for the Baynext ML application."""

import logging
import os
import ssl
import tempfile
from pathlib import Path
from typing import Literal

import mlflow
from meridian.analysis import analyzer
from meridian.data.load import CsvDataLoader, CoordToColumns
from meridian.mlflow import autolog
from meridian.model import model
from pydantic import BaseModel, ValidationError

# Set TensorFlow logging level to minimize verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Suppress XLA logs
os.environ["XLA_FLAGS"] = "--xla_hlo_profile=false"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# Disable SSL verification (not recommended for production)
if os.getenv("ENV", "dev") == "dev":
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


class BaynextLogger:
    """Custom logger for Baynext ML application."""

    def __init__(self) -> None:
        """Initialize BaynextLogger."""
        self.logger = logging.getLogger("baynext")
        self.run = mlflow.active_run()
        self.experiment_id = self.run.info.experiment_id
        self.run_name = self.run.info.run_name

    def _log(self, level: str, message: str) -> None:
        """Log a message at the specified level with experiment and run information."""
        log_message = f"{self.experiment_id} - {self.run_name} - {message}"
        if level == "info":
            self.logger.info(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "error":
            self.logger.error(log_message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self._log("info", message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._log("warning", message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self._log("error", message)


class LoadConfig(BaseModel):
    """Configuration for data loading."""

    csv_path: str | None = None
    kpi_type: Literal["non_revenue", "revenue"] = "non_revenue"
    coords_to_columns: CoordToColumns | None = None
    media_to_channel: dict[str, str] | None = None
    media_spend_to_channel: dict[str, str] | None = None


_N_DRAWS = 100
_N_CHAINS = 7
_N_ADAPT = 500
_N_BURNIN = 500
_N_KEEP = 1000


class PriorConfig(BaseModel):
    """Configuration for prior distributions."""

    n_draws: int = _N_DRAWS


class PosteriorConfig(BaseModel):
    """Configuration for posterior distributions."""

    n_chains: int = _N_CHAINS
    n_adapt: int = _N_ADAPT
    n_burnin: int = _N_BURNIN
    n_keep: int = _N_KEEP


class TrainConfig(BaseModel):
    """Configuration for training the model."""

    prior_config: PriorConfig = PriorConfig()
    posterior_config: PosteriorConfig = PosteriorConfig()


class AnalyzeConfig(BaseModel):
    """Configuration for analyzing the model."""


# TODO: use this https://ai.ragv.in/posts/sane-configs-with-pydantic-settings/
class PipelineConfig(BaseModel):
    """Configuration for the ML pipeline."""

    run_name: str | None = None
    description: str | None = None
    load_config: LoadConfig = LoadConfig()
    train_config: TrainConfig = TrainConfig()


def load(load_config: LoadConfig):
    """Load data based on the provided configuration."""


def train(train_config: TrainConfig):
    """Train the model based on the provided configuration."""


def analyze(analyze_config: AnalyzeConfig):
    """Analyze the model performance based on the provided configuration."""


def pipeline(pipeline_config: PipelineConfig) -> None:
    """Run the Meridian ML Training pipeline."""
    # Enable autologging (call this once per session)
    autolog.autolog(log_metrics=True)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    # Start an MLflow run (optionally name it for better grouping)
    with mlflow.start_run(
        run_name=pipeline_config.run_name,
        description=pipeline_config.description,
    ):
        logger = BaynextLogger()

        logger.info("⚡️ Starting Meridian ML pipeline")

        dataset_source = "https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_media.csv"

        # Load data
        coord_to_columns = CoordToColumns(
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

        loader = CsvDataLoader(
            csv_path=dataset_source,
            kpi_type=pipeline_config.load_config.kpi_type,
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

        logger.info("✅ Dataset loaded.")

        # Initialize Meridian model
        logger.info("🔧 Initializing Meridian model...")
        mmm = model.Meridian(input_data=data)

        # Run Meridian sampling processes
        logger.info("🔍 Running Meridian sampling processes...")
        mmm.sample_prior(
            n_draws=pipeline_config.train_config.prior_config.n_draws, seed=123
        )
        mmm.sample_posterior(
            n_chains=pipeline_config.train_config.posterior_config.n_chains,
            n_adapt=pipeline_config.train_config.posterior_config.n_adapt,
            n_burnin=pipeline_config.train_config.posterior_config.n_burnin,
            n_keep=pipeline_config.train_config.posterior_config.n_keep,
            seed=1,
        )

        logger.info("✅ Meridian sampling processes completed!")

        logger.info("🔄 Saving Meridian model...")
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = Path(tmpdirname) / "model.pkl"
            model.save_mmm(mmm, file_path)

            mlflow.log_artifact(file_path, "models")

            logger.info("✅ Meridian model saved!")

        logger.info("📊 Analyzing Meridian model...")
        a = analyzer.Analyzer(mmm)
        adstock_decay = a.adstock_decay()
        hill_curves = a.hill_curves()
        baseline_summary_metrics = a.baseline_summary_metrics().to_dataframe()
        summary_metrics = a.summary_metrics().to_dataframe()

        mlflow.log_table(adstock_decay, "adstock_decay.json")
        mlflow.log_table(hill_curves, "hill_curves.json")
        mlflow.log_table(baseline_summary_metrics, "baseline_summary_metrics.json")
        mlflow.log_table(summary_metrics, "summary_metrics.json")


if __name__ == "__main__":
    import argparse

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        prog="baynext-ml",
        description="Run the Meridian ML pipeline.",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Name of the MLflow run (If not provided, a random run name "
        "will be generated).",
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        required=True,
        help="Description of the MLflow run.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        help="Path to the input config file.",
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help=f"Number of draws for sampling (Default: {_N_DRAWS})",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=None,
        help=f"Number of chains for MCMC (Default: {_N_CHAINS}).",
    )
    parser.add_argument(
        "--adapt",
        type=int,
        default=None,
        help=f"Number of adaptation steps (Default: {_N_ADAPT}).",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=None,
        help=f"Number of burn-in steps (Default: {_N_BURNIN}).",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help=f"Number of samples to keep (Default: {_N_KEEP}).",
    )
    parser.add_argument(
        "--log-system-metrics",
        action="store_true",
        default=False,
        help="Enable MLflow system metrics logging.",
    )

    args = parser.parse_args()

    if args.file:
        import yaml

        with args.file.open("r") as file:
            try:
                pipeline_config = PipelineConfig(**yaml.safe_load(file))
            except ValidationError:
                logger.exception("Invalid configuration file")

    else:
        pipeline_config = PipelineConfig(
            run_name=args.name,
            description=args.message,
            train_config=TrainConfig(
                prior_config=PriorConfig(
                    n_draws=args.draws,
                ),
                posterior_config=PosteriorConfig(
                    n_chains=args.chains,
                    n_adapt=args.adapt,
                    n_burnin=args.burnin,
                    n_keep=args.keep,
                ),
            ),
        )

    if args.log_system_metrics:
        logger.info("Enabling MLflow system metrics logging.")
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = str(
            args.log_system_metrics,
        )

    pipeline(pipeline_config)
