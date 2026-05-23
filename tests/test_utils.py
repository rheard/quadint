import pytest

from quadint.utils import _is_squarefree  # noqa: PLC2701


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, False),
        (1, False),
        (-1, False),
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (6, True),
        (8, False),
        (9, False),
        (10, True),
        (12, False),
        (14, True),
        (15, True),
        (16, False),
        (18, False),
        (21, True),
        (22, True),
        (23, True),
        (24, False),
        (29, True),
        (31, True),
        (45, False),
        (61, True),
        (69, True),  # 69 = 3 * 23, squarefree
        (-14, True),
        (-18, False),
    ],
    ids=str,
)
def test_squarefree(n: int, *, expected: bool):
    """Verify the squarefree helper handles signs and repeated prime factors correctly."""
    assert _is_squarefree(n) is expected
