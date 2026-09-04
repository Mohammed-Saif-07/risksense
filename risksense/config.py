"""Central YAML configuration loader.

Every tunable value in RiskSense (window lengths, confidence levels, Monte
Carlo path counts, scenario definitions) lives in ``config/*.yaml`` — never
hard-coded. This module is the single point of access.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of the ``risksense`` package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


@functools.cache
def load_config(name: str, config_dir: str | None = None) -> dict[str, Any]:
    """Load a YAML config file by stem name (e.g. ``"model_params"``).

    Parameters
    ----------
    name:
        File stem under the config directory, without the ``.yaml`` suffix.
    config_dir:
        Optional override of the config directory (used by tests).

    Returns
    -------
    dict
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def data_dir() -> Path:
    """Return the repository ``data`` directory, creating subdirs on demand."""
    d = REPO_ROOT / "data"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "processed").mkdir(parents=True, exist_ok=True)
    return d
