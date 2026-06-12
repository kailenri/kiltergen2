"""Shared kinematics + dynamic-move helpers.

Used by both `sequence_generator.py` (beam search) and `lstm.py` (decode-time
masking) so the two code paths cannot drift on the geometric rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from config import (
    DYNAMIC_DNORM_TRIGGER,
    DYNAMIC_REACH_FACTOR,
    FLAG_LEG_FACTOR,
    FOREARM_LEN,
    HEAD_OFFSET,
    MAX_HAND_REACH,
    SHIN_LEN,
    THIGH_LEN,
    TORSO_LEN,
    UPPER_ARM_LEN,
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


# ---------------------------------------------------------------------------
# Articulated body model (2-bone IK for arms + legs)
# ---------------------------------------------------------------------------

def solve_2bone_ik(
    anchor_xy: XY,
    target_xy: XY,
    bone1_len: float,
    bone2_len: float,
    bend_dir: int = 1,
) -> Tuple[XY, XY]:
    """Solve 2-bone IK from anchor to target.

    Returns (joint_xy, end_effector_xy). If the target is beyond
    `bone1_len + bone2_len`, the chain points straight at the target and
    the end-effector is clamped to that maximum reach. If the target is
    inside `|bone1_len - bone2_len|`, the chain is folded and the end-
    effector is pushed out to the inner reach.

    `bend_dir`: +1 bends the joint to the LEFT of the anchor->target line
    (counter-clockwise), -1 bends to the RIGHT. Used to keep elbows behind
    the body and knees in front.
    """
    ax, ay = anchor_xy
    tx, ty = target_xy
    dx, dy = tx - ax, ty - ay
    d = math.hypot(dx, dy)

    L1, L2 = bone1_len, bone2_len
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)

    if d < 1e-9:
        # Anchor and target coincide: arbitrarily aim "down".
        return ((ax, ay - L1), (ax, ay - max_reach if max_reach > 0 else ay))

    if d >= max_reach:
        # Straight chain pointing at target; end-effector clamped.
        ux, uy = dx / d, dy / d
        joint = (ax + ux * L1, ay + uy * L1)
        end   = (ax + ux * max_reach, ay + uy * max_reach)
        return joint, end

    if d <= min_reach:
        # Folded chain; place joint on the far side of target along anchor->target.
        ux, uy = dx / d, dy / d
        joint = (ax + ux * L1, ay + uy * L1)
        end   = (ax + ux * min_reach, ay + uy * min_reach)
        return joint, end

    # Law of cosines: angle at anchor between (anchor->target) and (anchor->joint).
    cos_a = (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)
    cos_a = max(-1.0, min(1.0, cos_a))
    a = math.acos(cos_a)

    base = math.atan2(dy, dx)
    angle = base + (a if bend_dir >= 0 else -a)
    joint = (ax + L1 * math.cos(angle), ay + L1 * math.sin(angle))
    return joint, (tx, ty)


def _arm_bend_dir(limb: str, anchor_xy: XY, target_xy: XY) -> int:
    """Elbows should bend AWAY from the body centerline.

    For RH, anchor is the right shoulder; reaching out to the right, elbow
    bends right (away from spine). When reaching across, the elbow still
    prefers the outer side. Simple rule: bend in the direction matching the
    side of the body the hand is on (RH → +1 unless reaching far left).
    """
    side = 1 if limb == 'RH' else -1
    # If reaching far across (hand on opposite side of shoulder), flip.
    if (limb == 'RH' and target_xy[0] < anchor_xy[0] - X_SPACING * 0.5) or \
       (limb == 'LH' and target_xy[0] > anchor_xy[0] + X_SPACING * 0.5):
        side = -side
    return side


def _leg_bend_dir(limb: str, anchor_xy: XY, target_xy: XY) -> int:
    """Knees bend FORWARD (away from wall) which, in 2D board projection,
    we render as bending toward the body centerline so the leg silhouette
    has the characteristic 'frog' look when the foot is high.
    """
    side = -1 if limb == 'RF' else 1
    if (limb == 'RF' and target_xy[0] < anchor_xy[0] - X_SPACING * 0.5) or \
       (limb == 'LF' and target_xy[0] > anchor_xy[0] + X_SPACING * 0.5):
        side = -side
    return side


@dataclass
class Body:
    """Articulated body pose snapshot at one move."""
    hip: Optional[XY] = None
    shoulder: Optional[XY] = None
    head: Optional[XY] = None
    r_shoulder: Optional[XY] = None
    l_shoulder: Optional[XY] = None
    r_hip: Optional[XY] = None
    l_hip: Optional[XY] = None
    r_elbow: Optional[XY] = None
    l_elbow: Optional[XY] = None
    r_knee: Optional[XY] = None
    l_knee: Optional[XY] = None
    rh: Optional[XY] = None
    lh: Optional[XY] = None
    rf: Optional[XY] = None
    lf: Optional[XY] = None
    # Flag positions for feet that are flagging (off-hold, balancing in space).
    rf_flag: Optional[XY] = None
    lf_flag: Optional[XY] = None

    @classmethod
    def from_limb_positions(
        cls,
        limb_positions: Dict[str, Optional[XY]],
        cut_feet: Optional[Iterable[str]] = None,
        flags: Optional[Dict[str, Optional[XY]]] = None,
    ) -> 'Body':
        cut = set(cut_feet or [])
        flags = flags or {}

        # Effective foot positions: real hold OR flag-in-space (for COM/hip).
        rf = limb_positions.get('RF') if 'RF' not in cut else None
        lf = limb_positions.get('LF') if 'LF' not in cut else None
        rh = limb_positions.get('RH')
        lh = limb_positions.get('LH')
        rf_flag = flags.get('RF')
        lf_flag = flags.get('LF')

        # Hip = centroid of effective foot positions (planted OR flagging).
        effective_feet: List[XY] = []
        if rf is not None: effective_feet.append(rf)
        elif rf_flag is not None: effective_feet.append(rf_flag)
        if lf is not None: effective_feet.append(lf)
        elif lf_flag is not None: effective_feet.append(lf_flag)

        hip = estimate_hip(effective_feet) if effective_feet else None
        shoulder = estimate_shoulder(hip) if hip is not None else None
        head = (shoulder[0], shoulder[1] + HEAD_OFFSET) if shoulder is not None else None

        # Per-limb anchor offsets (~half-shoulder-width either side).
        half = X_SPACING * 0.5
        r_shoulder = (shoulder[0] + half, shoulder[1]) if shoulder is not None else None
        l_shoulder = (shoulder[0] - half, shoulder[1]) if shoulder is not None else None
        r_hip = (hip[0] + half, hip[1]) if hip is not None else None
        l_hip = (hip[0] - half, hip[1]) if hip is not None else None

        r_elbow = None
        l_elbow = None
        if rh is not None and r_shoulder is not None:
            r_elbow, _ = solve_2bone_ik(
                r_shoulder, rh, UPPER_ARM_LEN, FOREARM_LEN,
                bend_dir=_arm_bend_dir('RH', r_shoulder, rh),
            )
        if lh is not None and l_shoulder is not None:
            l_elbow, _ = solve_2bone_ik(
                l_shoulder, lh, UPPER_ARM_LEN, FOREARM_LEN,
                bend_dir=_arm_bend_dir('LH', l_shoulder, lh),
            )

        r_knee = None
        l_knee = None
        rf_target = rf if rf is not None else rf_flag
        lf_target = lf if lf is not None else lf_flag
        if rf_target is not None and r_hip is not None:
            r_knee, _ = solve_2bone_ik(
                r_hip, rf_target, THIGH_LEN, SHIN_LEN,
                bend_dir=_leg_bend_dir('RF', r_hip, rf_target),
            )
        if lf_target is not None and l_hip is not None:
            l_knee, _ = solve_2bone_ik(
                l_hip, lf_target, THIGH_LEN, SHIN_LEN,
                bend_dir=_leg_bend_dir('LF', l_hip, lf_target),
            )

        return cls(
            hip=hip, shoulder=shoulder, head=head,
            r_shoulder=r_shoulder, l_shoulder=l_shoulder,
            r_hip=r_hip, l_hip=l_hip,
            r_elbow=r_elbow, l_elbow=l_elbow,
            r_knee=r_knee, l_knee=l_knee,
            rh=rh, lh=lh, rf=rf, lf=lf,
            rf_flag=rf_flag, lf_flag=lf_flag,
        )


# ---------------------------------------------------------------------------
# Phantom flag rule
# ---------------------------------------------------------------------------

OPPOSITE_FOOT_FOR_HAND = {'RH': 'LF', 'LH': 'RF'}


def is_crossing_hand_move(
    new_hand_xy: XY,
    other_hand_xy: Optional[XY],
    limb: str,
    cross_pad: float = 0.5 * X_SPACING,
) -> bool:
    """Return True when the moving hand lands across the opposite hand.

    RH crossing means its new x is to the LEFT of LH (by `cross_pad`), and
    vice versa. Used as a trigger for flagging.
    """
    if other_hand_xy is None:
        return False
    if limb == 'RH':
        return new_hand_xy[0] < other_hand_xy[0] - cross_pad
    if limb == 'LH':
        return new_hand_xy[0] > other_hand_xy[0] + cross_pad
    return False


def compute_flag_position(
    hip_xy: XY,
    hand_target_xy: XY,
    flagging_foot: str,
) -> XY:
    """Compute the phantom (off-hold) position of a flagging foot.

    The flag extends the foot opposite to the hand reach direction, at a
    fraction of the total leg length, slightly below the hip. This gives
    the support polygon a counterweight on the opposite side.
    """
    leg_total = THIGH_LEN + SHIN_LEN
    reach = FLAG_LEG_FACTOR * leg_total
    hx, hy = hip_xy
    tx, ty = hand_target_xy
    dx, dy = tx - hx, ty - hy
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        # Degenerate: drop straight down on the opposite side.
        side = 1.0 if flagging_foot == 'RF' else -1.0
        return (hx + side * reach * 0.6, hy - reach * 0.6)
    ux, uy = dx / mag, dy / mag
    # Flag is opposite the hand reach, with a slight downward bias for stance.
    flag_x = hx - ux * reach
    flag_y = hy - uy * reach - 0.3 * X_SPACING
    return (flag_x, flag_y)


def choose_flag_foot(
    moving_hand: str,
    foot_positions: Dict[str, Optional[XY]],
) -> Optional[str]:
    """Pick which foot would flag for a given hand move.

    The opposite-side foot flags; only candidate if it is currently planted
    (cut or already-flagging feet are skipped — they're handled by the
    cut/replant path instead).
    """
    candidate = OPPOSITE_FOOT_FOR_HAND.get(moving_hand)
    if candidate and foot_positions.get(candidate) is not None:
        return candidate
    return None
