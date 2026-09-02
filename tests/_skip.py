# Copyright (c) 2026 Qingyuan Song
# SPDX-License-Identifier: MIT
"""Skip helper shared by the test modules.

These test files run two ways: under ``pytest``, and standalone as
``PYTHONPATH=src python3 tests/test_something.py``.  The historical pattern for
a test whose prerequisite is missing was::

    if not CORPUS.exists():
        print("SKIP: ...")
        return

which pytest scores as a **pass** — so a clean clone with no corpus reports
every test green even though a third of them did nothing.  ``skip()`` fixes
that without giving up the standalone runners: under pytest it raises, so the
test is reported as a skip with its reason; standalone it prints the same line
and lets the caller ``return``.

Usage::

    from _skip import skip
    ...
    if not CORPUS.exists():
        skip(f"needs a note corpus at {CORPUS}")
        return
"""

from __future__ import annotations

import os
import sys


def under_pytest() -> bool:
    """True while pytest is executing a test."""
    return "PYTEST_CURRENT_TEST" in os.environ


def skip(reason: str) -> None:
    """Abort the calling test as skipped (pytest) or print and continue.

    Under pytest this raises ``Skipped`` and never returns, so the ``return``
    that follows the call is unreachable there.  Standalone it prints and
    returns, and that ``return`` ends the test.
    """
    if under_pytest():
        import pytest

        pytest.skip(reason)
    caller = sys._getframe(1).f_code.co_name
    print(f"SKIP: {caller} ({reason})")
