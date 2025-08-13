"""Module to define Baynext ML settings."""

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from baynext.config.analyze import AnalyzeConfig
from baynext.config.load import LoadConfig
from baynext.config.log import LogConfig
from baynext.config.train import TrainConfig

_YAML_CONFIG_FILE = "baynext.yaml"


class PipelineConfig(BaseSettings):
    """Configuration for the ML pipeline."""

    run_name: str | None = None
    message: str = Field(description="Message to describe the run.")

    load: LoadConfig
    train: TrainConfig = TrainConfig()
    analyze: AnalyzeConfig = AnalyzeConfig()
    log: LogConfig = LogConfig()

    model_config = SettingsConfigDict(
        env_prefix="BAYNEXT_",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        json_file="config.json",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=_YAML_CONFIG_FILE,
            ),
            CliSettingsSource(
                settings_cls,
                cli_prog_name="baynext",
                cli_use_class_docs_for_groups=True,
                cli_parse_args=True,
                cli_hide_none_type=True,
                cli_avoid_json=True,
                cli_kebab_case=True,
                cli_shortcuts={
                    "run-name": "n",
                    "message": "m",
                },
            ),
        )
