"""Define the main CLI application for the ML module."""

import typer

app = typer.Typer(
    name="Baynext ML",
    help="Baynext Machine Learning Module CLI",
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.command()
def root() -> str:
    typer.echo("Hello from baynext-ml-cli!")
