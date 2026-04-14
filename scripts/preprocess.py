#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_retrieval.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["preprocess", *sys.argv[1:]]))
