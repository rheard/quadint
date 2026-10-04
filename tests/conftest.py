import faulthandler

from collections.abc import Iterator

import pytest


@pytest.fixture
def hang_guard() -> Iterator[None]:
    """
    End the test process if a test runs for minutes, so a loop that never stops fails fast instead of stalling CI.

    faulthandler prints every thread's traceback first, and it works even while compiled code holds the GIL. Under
        pytest-xdist only that worker dies, which fails the test it was running, and the rest carry on.
    """
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()
