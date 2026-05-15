"""Stable hashing for configs and run metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_hash(payload: Mapping[str, Any], *, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]

