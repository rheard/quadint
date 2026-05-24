from __future__ import annotations

from sympy import factorint


def _is_squarefree(n: int | dict[int, int]) -> bool:
    if isinstance(n, int):
        n = abs(n)
        if n <= 1:
            return False

        facts = factorint(n)
    else:
        facts = n

    return all(i < 2 for i in facts.values())
