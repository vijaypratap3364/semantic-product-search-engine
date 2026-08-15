"""Package foundation tests."""

from __future__ import annotations

import product_search


def test_package_import() -> None:
    assert product_search.__version__ == "0.1.0"
