"""Make source-checkout packages importable under every pytest launcher."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
root = str(REPOSITORY_ROOT)
if root not in sys.path:
    sys.path.insert(0, root)
