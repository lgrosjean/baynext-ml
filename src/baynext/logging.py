"""Defines logging config for Baynext training."""

import logging
from typing import Literal

import mlflow

# Set specific loggers to WARNING level to suppress INFO logs
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src.dispatch").setLevel(logging.WARNING)
logging.getLogger("jax._src.compiler").setLevel(logging.WARNING)

Step = Literal["load", "train", "analyze", "visualize"]


class BaynextLogger(logging.Logger):
    """Custom logger for Baynext ML application."""

    def __init__(
        self,
        step: Step = "load",
        log_level: int | str = logging.INFO,
        run: mlflow.ActiveRun | None = None,
    ) -> None:
        """Initialize BaynextLogger."""
        super().__init__("baynext")
        self.step = step
        self.run = run or mlflow.active_run()
        self.experiment_id = self.run.info.experiment_id
        self.run_name = self.run.info.run_name
        self.setLevel(log_level)

        if not self.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            file_handler = logging.FileHandler("baynext.log")
            file_handler.setFormatter(formatter)
            self.addHandler(stream_handler)
            self.addHandler(file_handler)

    @property
    def _prefix(self) -> str:
        """Return the log prefix for the current run."""
        return f"{self.experiment_id} [{self.run_name}][{self.step}] "

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Redefine logging method to include prefix."""
        super()._log(level, f"{self._prefix}{msg}", *args, **kwargs)

    def set_step(self, step: Step) -> None:
        """Set the current step for logging."""
        self.step = step

    def load(self) -> None:
        """Start the loading step."""
        self.set_step("load")
        self.info("🔄 Start loading dataset...")

    def train(self) -> None:
        """Start the training step."""
        self.set_step("train")
        self.info("🧠 Start Meridian model training...")

    def analyze(self) -> None:
        """Start the analysis step."""
        self.set_step("analyze")
        self.info("📊 Start analyzing model...")

    def visualize(self) -> None:
        """Start the visualization step."""
        self.set_step("visualize")
        self.info("📈 Start visualizing model...")
