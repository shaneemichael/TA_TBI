"""Configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nya_ir.exceptions import OptionalDependencyError


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON, TOML, or YAML config into a dictionary."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise OptionalDependencyError("tomli") from exc
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on installed env
            raise OptionalDependencyError("PyYAML") from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data or {}
    raise ValueError(f"Unsupported config format: {config_path}")
