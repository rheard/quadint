from __future__ import annotations

from sympy import factorint


def _is_squarefree(n: int) -> bool:
    n = abs(n)
    if n <= 1:
        return False

    facts = factorint(n)
    return all(i < 2 for i in facts.values())
