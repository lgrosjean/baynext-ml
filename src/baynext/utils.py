"""Utility fonctions for Baynext."""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from altair import Chart
from meridian.analysis.analyzer import Analyzer
from meridian.model import model

from baynext.config.load import SourceConfig

_MODEL_FILENAME = "model.pkl"
"""Default filename of the saved Meridian model"""
_MODEL_ARTIFACT_PATH = "models"
"""Default artifact path for the saved Meridian model in Mlflow Artifact store"""


def log_csv_dataset(url: str | Path, name: str | None = None) -> None:
    """Log the CSV dataset to MLflow."""
    dataset = mlflow.data.from_pandas(pd.read_csv(url), source=url, name=name)
    mlflow.log_input(dataset)


def log_dataset(source_config: SourceConfig) -> None:
    """Log the dataset to MLflow."""
    if source_config.type == "csv":
        return log_csv_dataset(source_config.path, name=source_config.name)

    msg = f"Logging for filetype {source_config.type} is not implemented."
    raise NotImplementedError(msg)


def log_model(mmm: model.Meridian) -> None:
    """Log the Meridian model to MLflow."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = Path(tmpdirname) / _MODEL_FILENAME
        model.save_mmm(mmm, file_path)
        mlflow.log_artifact(file_path, _MODEL_ARTIFACT_PATH)


def log_chart(chart: Chart, artifact_file: str, artifact_path: str = "plots") -> None:
    """Log an Altair chart to MLflow."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname) / artifact_file
        chart.save(filename, format="png")
        mlflow.log_artifact(filename, artifact_path)


def log_adstock_decay(analyzer: Analyzer) -> None:
    """Log the adstock decay table to MLflow."""
    mlflow.log_table(analyzer.adstock_decay(), "adstock_decay.json")


def log_hill_curves(analyzer: Analyzer) -> None:
    """Log the hill curves table to MLflow."""
    mlflow.log_table(analyzer.hill_curves(), "hill_curves.json")


def log_baseline_summary_metrics(analyzer: Analyzer) -> None:
    """Log the baseline summary metrics table to MLflow."""
    mlflow.log_table(
        analyzer.baseline_summary_metrics().to_dataframe(),
        "baseline_summary_metrics.json",
    )


def log_summary_metrics(analyzer: Analyzer) -> None:
    """Log the summary metrics table to MLflow."""
    mlflow.log_table(
        analyzer.summary_metrics().to_dataframe(),
        "summary_metrics.json",
    )
