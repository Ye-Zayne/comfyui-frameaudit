import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "frameaudit_test_package",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)


@pytest.fixture
def core():
    return sys.modules["frameaudit_test_package.core"]


@pytest.fixture
def package():
    return PACKAGE
