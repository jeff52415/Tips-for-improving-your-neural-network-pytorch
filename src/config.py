from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    model_name: str
    device: str
    label_smooth: Union[float, int]
    no_bias_decay: bool
    epochs: int
    warmup: bool
    mixup: bool
    lr: float
    num_classes: int
    distillation: bool
    stochastic_depth: bool
    group_normalization: bool
    weight_standardization: bool


def find_config_file() -> Path:
    """Locate the configuration file."""
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    raise Exception(f"Config not found at {DEFAULT_CONFIG_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None):
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path) as conf_file:
            parsed_config = yaml.load(conf_file, Loader=yaml.FullLoader)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config=None):
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = ModelConfig(**parsed_config)

    return _config


config = create_and_validate_config()
