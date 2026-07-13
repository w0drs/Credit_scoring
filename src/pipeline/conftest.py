"""
conftest.py
-----------
Pytest configuration for the ml-retraining-pipeline project.

This file adds the project root to sys.path so that `import pipeline.*`
works correctly regardless of where pytest is invoked from.

Place this file at the project root (same level as the pipeline/ package).
"""

import sys
from pathlib import Path

# Insert the project root so `from pipeline.xxx import ...` resolves correctly
# whether running: pytest, python -m pytest, or PyCharm's test runner.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
