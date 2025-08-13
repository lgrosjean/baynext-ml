"""Load Meridian data task."""

from meridian.data.input_data import InputData
from meridian.data.load import CoordToColumns, CsvDataLoader

from baynext.config.pipeline import LoadConfig


def load_task(load_config: LoadConfig) -> InputData:
    """Load data based on the provided configuration."""
    # Load data
    coord_to_columns = CoordToColumns(**load_config.coords_to_columns.model_dump())

    if load_config.source.type == "csv":
        return CsvDataLoader(
            csv_path=load_config.source.path,
            kpi_type=load_config.kpi_type,
            coord_to_columns=coord_to_columns,
            media_to_channel=load_config.media_to_channel,
            media_spend_to_channel=load_config.media_spend_to_channel,
        ).load()

    msg = f"Source type '{load_config.source.type}' is not supported."
    raise NotImplementedError(msg)
