from __future__ import annotations

import warnings

from quadint.quad.rings.base import (
    Factorization as Factorization,
    QuadraticRing as QuadraticRing,
    _check_den as _check_den,
    _choose_best_in_neighborhood as _choose_best_in_neighborhood,
    _NeighborhoodSearch as _NeighborhoodSearch,
    _round_div_ties_away_from_zero as _round_div_ties_away_from_zero,
    _split_uv as _split_uv,
)
from quadint.quad.rings.cornacchia import (
    CornacchiaRing as CornacchiaRing,
    EisensteinRing as EisensteinRing,
    GaussianRing as GaussianRing,
    SqrtMinusTwoRing as SqrtMinusTwoRing,
)
from quadint.quad.rings.harper import (
    Clark69Ring as Clark69Ring,
    HarperRing as HarperRing,
    _is_squarefree as _is_squarefree,
)
from quadint.quad.rings.heegner import (
    HeegnerDen2Ring as HeegnerDen2Ring,
    HeegnerElevenRing as HeegnerElevenRing,
    HeegnerNonEuclidUfdRing as HeegnerNonEuclidUfdRing,
    HeegnerSevenRing as HeegnerSevenRing,
)
from quadint.quad.rings.norm_euclid import (
    NORM_EUCLID_D as NORM_EUCLID_D,
    RealNormEuclidRing as RealNormEuclidRing,
)
from quadint.quad.rings.special import (
    DualRing as DualRing,
    SplitRing as SplitRing,
)

try:
    import cypari  # noqa: F401
except ImportError:
    warnings.warn(
        "cypari is not installed. "
        "Without it, Harper-like division is only available with rings that have a D value below 100 "
        "(because these are hard-coded).",
        ImportWarning,
        stacklevel=2,
    )
