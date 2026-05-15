"""Configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nya_ir.exceptions import OptionalDependencyError


def _ensure_mapping(data: Any, config_path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a top-level object: {config_path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON, TOML, or YAML config into a dictionary."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return _ensure_mapping(json.loads(config_path.read_text(encoding="utf-8")), config_path)
    if suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError as exc:  # pragma: no cover - depends on optional package
                raise OptionalDependencyError("tomli") from exc
        with config_path.open("rb") as handle:
            return _ensure_mapping(tomllib.load(handle), config_path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on installed env
            raise OptionalDependencyError("PyYAML") from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return _ensure_mapping(data or {}, config_path)
    raise ValueError(f"Unsupported config format: {config_path}")
