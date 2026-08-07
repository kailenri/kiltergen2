"""Hold metadata lookup for the angle-aware core.

Reads the same `data/hold_orientations.json` produced by
`scripts/build_hold_orientations.py` that the legacy generator uses, but
exposes it as a small cached accessor rather than module-level globals so the
core can be tested with injected data.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, Optional

from .params import DEFAULT_HOLD_QUALITY

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "hold_orientations.json",
)


@lru_cache(maxsize=4)
def _load(path: str) -> Dict[int, Dict]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return {}
    out: Dict[int, Dict] = {}
    for hid, entry in payload.get("holds", {}).items():
        try:
            out[int(hid)] = entry
        except (TypeError, ValueError):
            continue
    return out


class HoldInfo:
    """Quality and edge-orientation lookup, with a neutral fallback."""

    def __init__(self, path: Optional[str] = None, data: Optional[Dict[int, Dict]] = None):
        self._data = data if data is not None else _load(path or _DEFAULT_PATH)

    def quality(self, hold_id: Optional[int]) -> float:
        """Usability in [0, 1]. Neutral fallback for unmapped holds."""
        if hold_id is None:
            return DEFAULT_HOLD_QUALITY
        entry = self._data.get(hold_id)
        if entry is None:
            return DEFAULT_HOLD_QUALITY
        return float(entry.get("quality", DEFAULT_HOLD_QUALITY))

    def direction_deg(
        self, hold_id: Optional[int], min_confidence: float = 0.4
    ) -> Optional[float]:
        """Edge-tangent direction in degrees, or None when unknown.

        This is an undirected tangent: `d` and `d + 180` describe the same
        edge, which matters for how it is compared against a pull vector.
        """
        if hold_id is None:
            return None
        entry = self._data.get(hold_id)
        if entry is None:
            return None
        if float(entry.get("confidence", 0.0)) < min_confidence:
            return None
        return float(entry.get("direction_deg", 0.0))

    def pull_alignment(
        self, hold_id: Optional[int], pull_u: float, pull_v: float
    ) -> Optional[float]:
        """How well an in-plane pull direction suits this hold, in [0, 1].

        1.0 means pulling square against the hold's usable edge; 0.0 means
        pulling along it, where it gives nothing. Returns None when the hold
        has no confident orientation, so callers can skip the term rather
        than assume a value.
        """
        direction = self.direction_deg(hold_id)
        if direction is None:
            return None
        mag = math.hypot(pull_u, pull_v)
        if mag < 1e-9:
            return None

        # The usable pull direction is normal to the edge tangent.
        edge = math.radians(direction)
        normal = (-math.sin(edge), math.cos(edge))
        cos = abs((pull_u * normal[0] + pull_v * normal[1]) / mag)
        return max(0.0, min(1.0, cos))
