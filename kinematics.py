"""Shared kinematics + dynamic-move helpers.

Used by both `sequence_generator.py` (beam search) and `lstm.py` (decode-time
masking) so the two code paths cannot drift on the geometric rules.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from config import (
    DYNAMIC_DNORM_TRIGGER,
    DYNAMIC_REACH_FACTOR,
    MAX_HAND_REACH,
    X_SPACING,
)

XY = Tuple[float, float]


def estimate_hip(foot_positions: List[XY]) -> Optional[XY]:
    if not foot_positions:
        return None
    n = float(len(foot_positions))
    sx = sum(p[0] for p in foot_positions) / n
    sy = sum(p[1] for p in foot_positions) / n
    return (sx, sy)


def estimate_shoulder(hip_xy: XY) -> XY:
    return (hip_xy[0], hip_xy[1] + X_SPACING * 2.0)


def ellipse_dnorm(hold_xy: XY, shoulder_xy: XY, rx: float, ry: float) -> float:
    dx = (hold_xy[0] - shoulder_xy[0]) / rx
    dy = (hold_xy[1] - shoulder_xy[1]) / ry
    return math.hypot(dx, dy)


def convex_hull(points: Iterable[XY]) -> List[XY]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[XY] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[XY] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def point_in_poly(pt: XY, poly: List[XY]) -> bool:
    if not poly:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
        ):
            inside = not inside
    return inside


# ---------------------------------------------------------------------------
# Dynamic-move detection + cut-foot selection
# ---------------------------------------------------------------------------

OPPOSITE_FOOT = {'RH': 'LF', 'LH': 'RF'}


def is_dynamic_hand_move(
    prev_hand_xy: Optional[XY],
    new_hand_xy: XY,
    foot_positions: List[XY],
    moving_hand: str,
    other_hand_xy: Optional[XY] = None,
) -> bool:
    """Return True if this hand transition counts as a dynamic move.

    Triggers if EITHER:
      (a) Euclidean reach distance > DYNAMIC_REACH_FACTOR * MAX_HAND_REACH, OR
      (b) the estimated COM after the move falls outside the support polygon
          formed by the remaining planted limbs AND the move is non-trivially
          long (ellipse dnorm > DYNAMIC_DNORM_TRIGGER).
    """
    if prev_hand_xy is not None:
        dist = math.hypot(new_hand_xy[0] - prev_hand_xy[0],
                          new_hand_xy[1] - prev_hand_xy[1])
        if dist > DYNAMIC_REACH_FACTOR * MAX_HAND_REACH:
            return True

    if not foot_positions:
        return False

    hip = estimate_hip(foot_positions)
    if hip is None:
        return False
    shoulder = estimate_shoulder(hip)
    rx = MAX_HAND_REACH * 0.9
    ry = MAX_HAND_REACH * 0.6 + (X_SPACING * 1.2)
    dnorm = ellipse_dnorm(new_hand_xy, shoulder, rx, ry)
    if dnorm <= DYNAMIC_DNORM_TRIGGER:
        return False

    com = (hip[0], hip[1] + X_SPACING * 1.2)
    support = list(foot_positions)
    if other_hand_xy is not None:
        support.append(other_hand_xy)
    support.append(new_hand_xy)
    if len(support) < 3:
        return False
    hull = convex_hull(support)
    return not point_in_poly(com, hull)


def choose_cut_foot(
    moving_hand: str,
    foot_positions: Dict[str, Optional[XY]],
) -> Optional[str]:
    """Pick which foot is cut by a dynamic hand move.

    Heuristic: opposite-side foot (LH dyno → RF cuts, RH dyno → LF cuts).
    Falls back to whichever foot is planted; returns None if no foot is on.
    """
    preferred = OPPOSITE_FOOT.get(moving_hand)
    if preferred and foot_positions.get(preferred) is not None:
        return preferred
    for f in ('LF', 'RF'):
        if foot_positions.get(f) is not None:
            return f
    return None
