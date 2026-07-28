"""Scaffolding checks.

These exist so the repository is green from the first commit. Replace them
as soon as there is real physics to assert on.
"""

import tov_solver


def test_package_imports() -> None:
    assert tov_solver.__version__
