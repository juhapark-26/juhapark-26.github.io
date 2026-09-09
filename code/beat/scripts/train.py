#!/usr/bin/env python3
"""Train and evaluate one BEAT fold with the released paper implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

if __name__ == "__main__":
    from eccvw2.e2e_training import main

    main()
