#!/usr/bin/env python3
"""Run Stage 0–1 with a given config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsrag.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
