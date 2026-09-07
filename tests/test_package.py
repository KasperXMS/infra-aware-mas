"""Phase 0 package smoke tests."""

import infra_mas


def test_package_version() -> None:
    assert infra_mas.__version__ == "0.1.0"
