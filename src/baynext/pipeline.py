"""Main entry point for the Baynext ML application."""

import os

import mlflow
from meridian.analysis import analyzer, visualizer
from meridian.data.input_data import InputData
from meridian.mlflow import autolog
from meridian.model.model import Meridian

from baynext import tasks
from baynext.config import PipelineConfig
from baynext.config.pipeline import _YAML_CONFIG_FILE
from baynext.logging import BaynextLogger
from baynext.utils import (
    log_adstock_decay,
    log_baseline_summary_metrics,
    log_chart,
    log_dataset,
    log_hill_curves,
    log_model,
    log_summary_metrics,
)


class Pipeline:
    """Pipeline for Baynext ML application."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        """Initialize the pipeline."""
        self.config = pipeline_config
        self.logger = BaynextLogger()

        self.data_ = None
        self.model_ = None

    def log_config(self) -> None:
        """Log the pipeline configuration."""
        mlflow.log_dict(self.config.model_dump(), artifact_file=_YAML_CONFIG_FILE)

    def load(self) -> None:
        """Run the data loading step of the pipeline."""
        self.logger.load()
        self.data_ = tasks.load(self.config.load)
        self.logger.info("✅ Dataset loaded.")

        if self.config.log.dataset:
            self.logger.info("🔄 Saving Meridian dataset...")
            log_dataset(self.config.load.source)
            self.logger.info("✅ Meridian dataset saved.")

    def train(self) -> None:
        """Run the training step of the pipeline."""
        self.logger.train()
        if not self.data_:
            raise ValueError

        self.model_ = tasks.train(
            input_data=self.data_,
            train_config=self.config.train,
        )
        self.logger.info("✅ Meridian model training completed.")

        if self.config.log.model:
            self.logger.info("🔄 Saving Meridian model...")
            log_model(self.model_)
            self.logger.info("✅ Meridian model saved!")

    def analyze(self) -> None:
        """Run the analysis step of the pipeline."""
        if not self.model_:
            raise ValueError

        self.logger.analyze()
        a = analyzer.Analyzer(self.model_)

        log_adstock_decay(a)
        self.logger.info("✅ Adstock decay table logged.")

        log_hill_curves(a)
        self.logger.info("✅ Hill curves table logged.")

        log_baseline_summary_metrics(a)
        self.logger.info("✅ Baseline summary metrics table logged.")

        log_summary_metrics(a)
        self.logger.info("✅ Summary metrics table logged.")

    def visualize(self) -> None:
        """Run the visualization step of the pipeline."""
        if not self.model_:
            raise ValueError

        self.logger.visualize()
        model_fit = visualizer.ModelFit(self.model_)

        log_chart(
            model_fit.plot_model_fit(),
            "model_fit.png",
        )
        self.logger.info("✅ Model fit chart logged.")

        media_summary = visualizer.MediaSummary(self.model_)

        mlflow.log_table(
            media_summary.summary_table(),
            "media_summary.json",
        )
        self.logger.info("✅ Media summary table logged.")

        log_chart(
            media_summary.plot_channel_contribution_area_chart(),
            "channel_contribution_area_chart.png",
        )
        self.logger.info("✅ Channel contribution area chart logged.")

        log_chart(
            media_summary.plot_contribution_waterfall_chart(),
            "contribution_waterfall_chart.png",
        )
        self.logger.info("✅ Contribution waterfall chart logged.")

        log_chart(
            media_summary.plot_contribution_pie_chart(),
            "contribution_pie_chart.png",
        )
        self.logger.info("✅ Contribution pie chart logged.")

        log_chart(
            media_summary.plot_spend_vs_contribution(),
            "spend_vs_contribution.png",
        )
        self.logger.info("✅ Spend vs Contribution chart logged.")

        log_chart(
            media_summary.plot_roi_bar_chart(),
            "roi_bar_chart.png",
        )
        self.logger.info("✅ ROI bar chart logged.")

    def run(self) -> None:
        """Run the entire pipeline."""
        self.logger.info("⚡️ Starting Baynext ML pipeline")
        try:
            self.load()
        except Exception:
            self.logger.exception("❌ Error occurred while loading dataset.")
            raise

        try:
            self.train()
        except Exception:
            self.logger.exception("❌ Error occurred while training model.")
            raise

        try:
            self.analyze()
        except Exception:
            self.logger.exception("❌ Error occurred while analyzing model.")
            raise

        try:
            self.visualize()
        except Exception:
            self.logger.exception("❌ Error occurred while visualizing model.")
            raise

        self.logger.info("✅ Baynext ML pipeline ended")


def run_pipeline(pipeline_config: PipelineConfig) -> None:
    """Run the Meridian ML Training pipeline."""
    # Enable autologging (call this once per session)
    autolog.autolog(log_metrics=pipeline_config.log.metrics)

    # Set the MLflow tracking URI, default to localhost
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    # Start an MLflow run (optionally name it for better grouping)
    with mlflow.start_run(
        run_name=pipeline_config.run_name,
        description=pipeline_config.message,
    ):
        pipeline = Pipeline(pipeline_config)
        pipeline.log_config()
        pipeline.run()
