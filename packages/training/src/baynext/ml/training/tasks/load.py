"""Module to load input data for Meridian model training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from meridian.data.load import CoordToColumns, CsvDataLoader

if TYPE_CHECKING:
    from meridian.data.input_data import InputData

    from utils.enums import KPIType


from training.logger import get_logger

logger = get_logger(__name__)

MEDIA_TO_CHANNEL = {
    "Channel0_impression": "Channel_0",
    "Channel1_impression": "Channel_1",
    "Channel2_impression": "Channel_2",
    "Channel3_impression": "Channel_3",
    "Channel4_impression": "Channel_4",
}
MEDIA_SPEND_TO_CHANNEL = {
    "Channel0_spend": "Channel_0",
    "Channel1_spend": "Channel_1",
    "Channel2_spend": "Channel_2",
    "Channel3_spend": "Channel_3",
    "Channel4_spend": "Channel_4",
}


def task(
    csv_path: str,
    kpi_type: str,
    time: str,
    kpi: KPIType,
    controls: list[str],
    geo: str,
    population: str,
    revenue_per_kpi: str | None = None,
    media: list[str] | None = None,
    media_spend: list[str] | None = None,
    organic_media: list[str] | None = None,
    non_media_treatments: list[str] | None = None,
    media_to_channel: dict[str, str] = MEDIA_TO_CHANNEL,
    media_spend_to_channel: dict[str, str] = MEDIA_SPEND_TO_CHANNEL,
) -> InputData:
    """Load input data from a CSV file for Meridian model training."""
    coord_to_columns = CoordToColumns(
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

    logger.info("Time: %s", time)
    logger.info("KPI: %s", kpi)
    logger.info("Controls: %s", controls)
    logger.info("Geo: %s", geo)
    logger.info("Population: %s", population)
    logger.info("Revenue per KPI: %s", revenue_per_kpi)
    logger.info("Media: %s", media)
    logger.info("Media Spend: %s", media_spend)
    logger.info("Organic Media: %s", organic_media)
    logger.info("Non-media Treatments: %s", non_media_treatments)

    loader = CsvDataLoader(
        csv_path=csv_path,
        kpi_type=kpi_type,
        coord_to_columns=coord_to_columns,
        media_to_channel=media_to_channel,
        media_spend_to_channel=media_spend_to_channel,
    )

    logger.info("KPI Type: %s", kpi_type)
    logger.info("Media to Channel: %s", media_to_channel)
    logger.info("Media Spend to Channel: %s", media_spend_to_channel)

    return loader.load()
