"""CLI for Meridian model training pipeline.

This module provides a command-line interface using Typer to run the Meridian
model training pipeline with configurable parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from training.main import main as run_training_pipeline
from utils.constants import (
    MAX_LAG,
    N_ADAPT,
    N_BURNIN,
    N_CHAINS,
    N_DRAWS,
    N_KEEP,
    OUTPUT_FILENAME,
    ROI_MU,
    ROI_SIGMA,
)
from utils.enums import KPIType  # noqa: TC001

app = typer.Typer()
console = Console()


@app.command()
def train(
    csv_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the CSV file containing input data",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    kpi_type: Annotated[
        KPIType,
        typer.Option(
            "--kpi-type",
            help="Type of KPI to analyze (e.g., 'revenue', 'conversions')",
        ),
    ],
    time: Annotated[
        str,
        typer.Option(
            "--time",
            help="Name of the time column in the data",
        ),
    ],
    kpi: Annotated[
        str,
        typer.Option(
            "--kpi",
            help="Name of the KPI column in the data",
        ),
    ],
    controls: Annotated[
        str,
        typer.Option(
            "--controls",
            help="Comma-separated list of control variables",
        ),
    ],
    geo: Annotated[
        str,
        typer.Option(
            "--geo",
            help="Name of the geographic variable column",
        ),
    ],
    population: Annotated[
        str,
        typer.Option(
            "--population",
            help="Name of the population variable column",
        ),
    ],
    revenue_per_kpi: Annotated[
        str | None,
        typer.Option(
            "--revenue-per-kpi",
            help="Name of the revenue per KPI column (optional)",
        ),
    ] = None,
    media: Annotated[
        str | None,
        typer.Option(
            "--media",
            help="Comma-separated list of media channels (optional)",
        ),
    ] = None,
    media_spend: Annotated[
        str | None,
        typer.Option(
            "--media-spend",
            help="Comma-separated list of media spend channels (optional)",
        ),
    ] = None,
    organic_media: Annotated[
        str | None,
        typer.Option(
            "--organic-media",
            help="Comma-separated list of organic media channels (optional)",
        ),
    ] = None,
    non_media_treatments: Annotated[
        str | None,
        typer.Option(
            "--non-media-treatments",
            help="Comma-separated list of non-media treatments (optional)",
        ),
    ] = None,
    roi_mu: Annotated[
        float,
        typer.Option(
            "--roi-mu",
            help="Mean of the ROI prior distribution",
            min=0.0,
        ),
    ] = ROI_MU,
    roi_sigma: Annotated[
        float,
        typer.Option(
            "--roi-sigma",
            help="Standard deviation of the ROI prior distribution",
            min=0.0,
        ),
    ] = ROI_SIGMA,
    max_lag: Annotated[
        int,
        typer.Option(
            "--max-lag",
            help="Maximum lag for the model",
            min=1,
        ),
    ] = MAX_LAG,
    n_draws: Annotated[
        int,
        typer.Option(
            "--n-draws",
            help="Number of draws for prior sampling",
            min=100,
        ),
    ] = N_DRAWS,
    n_chains: Annotated[
        int,
        typer.Option(
            "--n-chains",
            help="Number of chains for posterior sampling",
            min=1,
        ),
    ] = N_CHAINS,
    n_adapt: Annotated[
        int,
        typer.Option(
            "--n-adapt",
            help="Number of adaptation steps for MCMC sampling",
            min=100,
        ),
    ] = N_ADAPT,
    n_burnin: Annotated[
        int,
        typer.Option(
            "--n-burnin",
            help="Number of burn-in steps for MCMC sampling",
            min=100,
        ),
    ] = N_BURNIN,
    n_keep: Annotated[
        int,
        typer.Option(
            "--n-keep",
            help="Number of samples to keep after burn-in",
            min=100,
        ),
    ] = N_KEEP,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to save the trained model",
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path(OUTPUT_FILENAME),
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    r"""🚀 Meridian model training pipeline CLI.

    This command loads data from a CSV file, prepares the model specification,
    trains the Meridian model, and saves it to the specified output path.

    Example usage:
        meridian-training train data.csv \
            --kpi-type revenue \
            --time week \
            --kpi sales \
            --controls "trend,seasonality" \
            --geo region \
            --population population \
            --media "tv,radio,digital" \
            --media-spend "tv_spend,radio_spend,digital_spend" \
            --output my_model.pkl
    """
    if verbose:
        console.print("🔧 [bold blue]Configuration:[/bold blue]")
        console.print(f"  📁 Input file: {csv_path}")
        console.print(f"  📊 KPI type: {kpi_type}")
        console.print(f"  ⏰ Time column: {time}")
        console.print(f"  🎯 KPI column: {kpi}")
        console.print(f"  🎛️  Controls: {controls}")
        console.print(f"  🌍 Geography: {geo}")
        console.print(f"  👥 Population: {population}")
        console.print(f"  💰 Revenue per KPI: {revenue_per_kpi or 'None'}")
        console.print(f"  📺 Media channels: {media or 'None'}")
        console.print(f"  💸 Media spend: {media_spend or 'None'}")
        console.print(f"  🌱 Organic media: {organic_media or 'None'}")
        console.print(f"  🔧 Non-media treatments: {non_media_treatments or 'None'}")
        console.print(f"  📈 ROI mu: {roi_mu}")
        console.print(f"  📊 ROI sigma: {roi_sigma}")
        console.print(f"  ⏳ Max lag: {max_lag}")
        console.print(f"  🎲 Draws: {n_draws}")
        console.print(f"  🔗 Chains: {n_chains}")
        console.print(f"  🔄 Adaptation steps: {n_adapt}")
        console.print(f"  🔥 Burn-in steps: {n_burnin}")
        console.print(f"  💾 Samples to keep: {n_keep}")
        console.print(f"  📤 Output file: {output}")
        console.print()

    try:
        # Parse comma-separated lists
        controls_list = [c.strip() for c in controls.split(",") if c.strip()]
        media_list = (
            [m.strip() for m in media.split(",") if m.strip()] if media else None
        )
        media_spend_list = (
            [ms.strip() for ms in media_spend.split(",") if ms.strip()]
            if media_spend
            else None
        )
        organic_media_list = (
            [om.strip() for om in organic_media.split(",") if om.strip()]
            if organic_media
            else None
        )
        non_media_treatments_list = (
            [nmt.strip() for nmt in non_media_treatments.split(",") if nmt.strip()]
            if non_media_treatments
            else None
        )

        # Validate that media and media_spend have same length if both provided
        if media_list and media_spend_list and len(media_list) != len(media_spend_list):
            console.print(
                "❌ [bold red]Error:[/bold red] Media channels and media spend lists",
                "must have the same length",
                style="red",
            )
            raise typer.Exit(1) from None

        # Run the training pipeline
        run_training_pipeline(
            csv_path=str(csv_path),
            kpi_type=kpi_type,
            time=time,
            kpi=kpi,
            controls=controls_list,
            geo=geo,
            population=population,
            revenue_per_kpi=revenue_per_kpi,
            media=media_list,
            media_spend=media_spend_list,
            organic_media=organic_media_list,
            non_media_treatments=non_media_treatments_list,
            roi_mu=roi_mu,
            roi_sigma=roi_sigma,
            max_lag=max_lag,
            n_draws=n_draws,
            n_chains=n_chains,
            n_adapt=n_adapt,
            n_burnin=n_burnin,
            n_keep=n_keep,
            file_path=str(output),
        )

        console.print("✅ [bold green]Training completed successfully![/bold green]")
        console.print(f"📁 Model saved to: {output}")

    except (ValueError, FileNotFoundError, KeyError) as e:
        console.print(f"❌ [bold red]Training failed:[/bold red] {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e
