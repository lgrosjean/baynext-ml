"""Configuration for loading the model."""

from typing import Literal

from meridian.data.load import constants
from pydantic import BaseModel

KpiType = Literal["non_revenue", "revenue"]


class CoordToCols(BaseModel):
    """A mapping between the desired and actual column names in the input data."""

    time: str = constants.TIME
    """Name of column containing `time` values."""
    geo: str = constants.GEO
    """Name of column containing `geo` values.
    This field is optional for a national model."""
    kpi: str = constants.KPI
    """Name of column containing `kpi` values."""
    controls: list[str] | None = None
    """List of column names containing `controls` values. Optional."""
    revenue_per_kpi: str | None = None
    """Name of column containing `revenue_per_kpi` values. Optional.
    Will be overridden if model KPI type is `revenue`."""
    population: str = constants.POPULATION
    # Media data
    media: list[str] | None = None
    media_spend: list[str] | None = None
    # RF data
    reach: list[str] | None = None
    frequency: list[str] | None = None
    rf_spend: list[str] | None = None
    # Non-media treatments data
    non_media_treatments: list[str] | None = None
    # Organic media and RF data
    organic_media: list[str] | None = None
    organic_reach: list[str] | None = None
    organic_frequency: list[str] | None = None


SourceType = Literal["csv"]


class SourceConfig(BaseModel):
    """Configuration for the data source."""

    type: SourceType = "csv"
    """The type of the data source."""
    name: str | None = None
    """Name of the data source."""
    path: str
    """The path to the data source."""


class LoadConfig(BaseModel):
    """Configuration for data loading."""

    source: SourceConfig
    """The configuration for the data source."""
    kpi_type: KpiType = "non_revenue"
    coords_to_columns: CoordToCols
    """A CoordToColumns object whose fields are the desired coordinates of the InputData
    and the values are the current names of columns (or lists of columns)
    in the CSV file."""
    media_to_channel: dict[str, str] | None = None
    """A dictionary whose keys are the actual column names for `media` data
    in the CSV file and values are the desired channel names,
    the same as for the media_spend data."""
    media_spend_to_channel: dict[str, str] | None = None
    """A dictionary whose keys are the actual column names for `media_spend` data in the
    CSV file and values are the desired channel names, the same as for the media data.
    """
    reach_to_channel: dict[str, str] | None = None
    frequency_to_channel: dict[str, str] | None = None
    rf_spend_to_channel: dict[str, str] | None = None
    organic_reach_to_channel: dict[str, str] | None = None
    organic_frequency_to_channel: dict[str, str] | None = None
