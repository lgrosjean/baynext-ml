"""Main entry point for the Baynext ML application."""

import os

import mlflow
from meridian.analysis import analyzer, visualizer
from meridian.data.input_data import InputData
from meridian.mlflow import autolog
from meridian.model.model import Meridian

from baynext import tasks
from baynext.logging import BaynextLogger
from baynext.config.pipeline import PipelineConfig
from baynext.utils import (
    log_adstock_decay,
    log_baseline_summary_metrics,
    log_chart,
    log_dataset,
    log_hill_curves,
    log_model,
    log_summary_metrics,
)


def run_load(logger: BaynextLogger, pipeline_config: PipelineConfig) -> None:
    """Run the data loading step of the pipeline."""
    logger.load()
    data = tasks.load(pipeline_config.load)
    logger.info("✅ Dataset loaded.")

    if pipeline_config.log.dataset:
        logger.info("🔄 Saving Meridian dataset...")
        log_dataset(pipeline_config.load.source)
        logger.info("✅ Meridian dataset saved.")

    return data


def run_train(
    logger: BaynextLogger,
    data: InputData,
    pipeline_config: PipelineConfig,
) -> None:
    """Run the model training step of the pipeline."""
    logger.train()
    mmm = tasks.train(
        input_data=data,
        train_config=pipeline_config.train,
    )
    logger.info("✅ Meridian model training completed.")

    if pipeline_config.log.model:
        logger.info("🔄 Saving Meridian model...")
        log_model(mmm)
        logger.info("✅ Meridian model saved!")

    return mmm


def run_analyze(logger: BaynextLogger, mmm: Meridian) -> None:
    """Run the model analysis step of the pipeline."""
    logger.analyze()
    a = analyzer.Analyzer(mmm)

    log_adstock_decay(a)
    logger.info("✅ Adstock decay table logged.")

    log_hill_curves(a)
    logger.info("✅ Hill curves table logged.")

    log_baseline_summary_metrics(a)
    logger.info("✅ Baseline summary metrics table logged.")

    log_summary_metrics(a)
    logger.info("✅ Summary metrics table logged.")


def run_visualize(logger: BaynextLogger, mmm: Meridian) -> None:
    """Run the model visualization step of the pipeline."""
    logger.visualize()
    model_fit = visualizer.ModelFit(mmm)

    log_chart(
        model_fit.plot_model_fit(),
        "model_fit.png",
    )
    logger.info("✅ Model fit chart logged.")

    media_summary = visualizer.MediaSummary(mmm)

    mlflow.log_table(
        media_summary.summary_table(),
        "media_summary.json",
    )
    logger.info("✅ Media summary table logged.")

    log_chart(
        media_summary.plot_channel_contribution_area_chart(),
        "channel_contribution_area_chart.png",
    )
    logger.info("✅ Channel contribution area chart logged.")

    log_chart(
        media_summary.plot_contribution_waterfall_chart(),
        "contribution_waterfall_chart.png",
    )
    logger.info("✅ Contribution waterfall chart logged.")

    log_chart(
        media_summary.plot_contribution_pie_chart(),
        "contribution_pie_chart.png",
    )
    logger.info("✅ Contribution pie chart logged.")

    log_chart(
        media_summary.plot_spend_vs_contribution(),
        "spend_vs_contribution.png",
    )
    logger.info("✅ Spend vs Contribution chart logged.")

    log_chart(
        media_summary.plot_roi_bar_chart(),
        "roi_bar_chart.png",
    )
    logger.info("✅ ROI bar chart logged.")


def pipeline(pipeline_config: PipelineConfig) -> None:
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
        logger = BaynextLogger()

        # Load
        try:
            data = run_load(logger, pipeline_config)
        except Exception:
            logger.exception("❌ Error occurred while loading dataset.")
            raise

        # Train
        try:
            mmm = run_train(logger, data, pipeline_config)
        except Exception:
            logger.exception("❌ Error occurred during model training")
            raise

        # Analyze
        try:
            run_analyze(logger, mmm)
        except Exception:
            logger.exception("❌ Error occurred during model analysis")
            raise

        # Visualize
        try:
            run_visualize(logger, mmm)
        except Exception:
            logger.exception("❌ Error occurred during model visualization")
            raise

        logger.end()
