"""Repository wrapper for ``nya_ir.cli.evaluate_runs``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nya_ir.cli.evaluate_runs import main


if __name__ == "__main__":
    raise SystemExit(main())

