"""Puts the repository root on `sys.path` for pytest.

Scripts and library modules import as `src.*` / `experiments.*`, which assumes
the repository root is importable. A test file has its own directory prepended
by pytest, not the root, so without this the test suite could only be run
through `python -m pytest` from exactly the right directory. A `conftest.py`
here makes the root importable however pytest is invoked.
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
