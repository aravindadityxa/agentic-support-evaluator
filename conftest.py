"""
Pytest configuration - prevents hanging on collection

Adds project root to sys.path and configures pytest.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent pytest from trying to import modules that hang
pytest_plugins = []
