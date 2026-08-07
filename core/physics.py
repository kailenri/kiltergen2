"""Angle-aware statics for board climbing.

The existing `kinematics.py` works entirely in the wall plane, so it cannot
distinguish a 20-degree board from a 60-degree one -- the same holds in the
same order produce identical geometry. This module supplies the missing
dimension: where gravity points relative to the wall.

Frame
-----
Board coordinates (x, y) live in the wall plane, y running up the wall. The
wall overhangs by `angle` degrees past vertical. Two unit vectors matter:

    u_hat  -- up the wall, in-plane (this is +y in board coordinates)
    n_hat  -- the wall normal, pointing away from the surface toward the
              climber (out of the board plane)

Gravity, per unit body weight, decomposes as::

    g = -cos(angle) * u_hat  +  sin(angle) * n_hat

At 0 degrees (vertical) the whole weight runs down the wall and nothing pulls
the climber off it. At 90 degrees (roof) nothing runs down the wall and the
entire weight pulls straight off. Every angle-dependent behaviour in the cost
model falls out of that one split.

Statics
-------
We reduce to the sagittal plane (the plane spanned by u_hat and n_hat) and
solve a two-contact-group problem: hands at mean height `u_h` on the wall
surface, feet at mean height `u_f` on the surface, centre of mass at height
`u_c` and standing `COM_WALL_OFFSET` out from the surface.

Normal-direction force balance gives the pull the hands must supply::

    F_hand_n = W * sin(angle) + F_foot_n

Moment balance about the hands gives what the feet must supply normal to the
wall to hold the pose::

    F_foot_n = W * (d_com * cos(angle) - a * sin(angle)) / L

where ``d_com`` is the COM's distance from the wall, ``a`` is how far the COM
sits below the hands, and ``L`` is the hand-to-foot span. A *positive* value
means the wall is pressing back on the feet -- the climber is standing on
them. A *negative* value means the feet would have to pull inward to hold the
pose, which a standard foothold cannot do: the climber must generate that
moment internally with core tension, or the feet swing off.

That sign change is the whole story of board angle. It happens at::

    tan(angle) = d_com / a

which for a typical pose lands around 25-30 degrees, and past it the cost of
keeping feet on is a real, computable number rather than a tuned constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from kinematics import convex_hull, point_in_poly

from .params import (
    COM_WALL_OFFSET,
    FOOT_EDGE_CAPACITY,
    FOOT_SMEAR_RESIDUAL,
    MAX_CORE_TENSION,
    MIN_CONTACT_SPAN,
)

XY = Tuple[float, float]


def gravity_components(angle_deg: float) -> Tuple[float, float]:
    """Split unit gravity into (down-the-wall, off-the-wall) components.

    Returns ``(parallel, normal)`` as positive magnitudes per unit body
    weight: `parallel` runs down `-u_hat`, `normal` runs out along `+n_hat`.
    """
    theta = math.radians(angle_deg)
    return (math.cos(theta), math.sin(theta))


@dataclass(frozen=True)
class ContactLoads:
    """Resolved forces for one body pose at one board angle.

    All magnitudes are fractions of body weight.
    """

    angle_deg: float
    #: Inward pull the hands must supply to stay on the wall.
    hand_normal: float
    #: Down-wall load the hands must carry (what the feet could not take).
    hand_shear: float
    #: Outward force the wall exerts on the feet. Zero once the pose needs
    #: the feet to pull rather than press.
    foot_normal: float
    #: Down-wall load actually carried by the feet.
    foot_shear: float
    #: Core moment demand, as a fraction of body weight, needed to keep the
    #: feet on. Zero on slab, grows with angle and with body extension.
    tension_demand: float
    #: How strongly the feet are working, in [0, 1]. 1.0 means standing on
    #: them, 0.0 means they contribute nothing.
    foot_engagement: float
    #: False when `tension_demand` exceeds what the core can generate -- the
    #: pose is unholdable and the feet cut whether the climber likes it
    #: or not.
    feet_secure: bool
    #: True when the COM projects inside the support polygon of the planted
    #: contacts.
    balanced: bool

    @property
    def hand_load(self) -> float:
        """Total force through the hands, as a fraction of body weight."""
        return math.hypot(self.hand_normal, self.hand_shear)


def _mean_y(points: Sequence[XY]) -> Optional[float]:
    if not points:
        return None
    return sum(p[1] for p in points) / float(len(points))


def solve_contact_loads(
    hand_xys: Sequence[XY],
    foot_xys: Sequence[XY],
    com_xy: XY,
    angle_deg: float,
    foot_quality: float = 1.0,
) -> ContactLoads:
    """Resolve the forces holding one pose on a board at `angle_deg`.

    `hand_xys` and `foot_xys` are the wall-plane positions of the contacts
    currently loaded; feet that are cut should be omitted, feet that are
    flagging should be passed at their phantom position (a flag is a real
    stance and does contribute to balance, though it carries little load).

    `foot_quality` in [0, 1] scales how much down-wall load the footholds can
    accept -- a sloping smear takes less than a positive edge.
    """
    g_par, g_norm = gravity_components(angle_deg)

    u_h = _mean_y(hand_xys)
    u_f = _mean_y(foot_xys)

    support = list(hand_xys) + list(foot_xys)
    balanced = _com_in_support(com_xy, support)

    # No feet at all: hands carry everything, and there is nothing to keep on.
    if u_f is None or u_h is None:
        return ContactLoads(
            angle_deg=angle_deg,
            hand_normal=g_norm,
            hand_shear=g_par,
            foot_normal=0.0,
            foot_shear=0.0,
            tension_demand=0.0,
            foot_engagement=0.0,
            feet_secure=False,
            balanced=balanced,
        )

    # Moment arms. `span` is hand-to-foot separation up the wall; `com_below`
    # is how far the COM hangs beneath the hands. Both are clamped so a
    # scrunched pose degrades smoothly instead of dividing by ~zero.
    span = max(u_h - u_f, MIN_CONTACT_SPAN)
    com_below = max(u_h - com_xy[1], 0.0)

    # Moment balance about the hands.
    required_foot_normal = (
        COM_WALL_OFFSET * g_par - com_below * g_norm
    ) / span

    if required_foot_normal >= 0.0:
        # The wall presses back on the feet: the climber is standing on them.
        foot_normal = required_foot_normal
        tension_demand = 0.0
        engagement = 1.0
        feet_secure = True
    else:
        # The pose wants the feet to pull inward. A foothold cannot, so the
        # core has to supply the moment. What it cannot supply is a foot cut.
        demand = -required_foot_normal
        foot_normal = 0.0
        tension_demand = demand
        feet_secure = demand <= MAX_CORE_TENSION
        engagement = max(0.0, 1.0 - demand / MAX_CORE_TENSION)

    # Feet take down-wall load as long as they are pressed into the wall.
    # Note this is gated on security and engagement but deliberately *not* on
    # balance: a climber whose COM has swung outside the support polygon is
    # barn-dooring and needs the hands to stop the rotation, but their feet
    # are still standing on the holds. Folding balance in here made the feet
    # contribute nothing at any angle, which inverted the whole angle trend.
    # Balance is priced separately, as its own cost term.
    if feet_secure:
        capacity = FOOT_EDGE_CAPACITY * foot_quality * engagement
    else:
        capacity = FOOT_SMEAR_RESIDUAL * foot_quality
    foot_shear = g_par * min(1.0, capacity)

    hand_shear = g_par - foot_shear
    hand_normal = g_norm + foot_normal

    return ContactLoads(
        angle_deg=angle_deg,
        hand_normal=hand_normal,
        hand_shear=hand_shear,
        foot_normal=foot_normal,
        foot_shear=foot_shear,
        tension_demand=tension_demand,
        foot_engagement=engagement,
        feet_secure=feet_secure,
        balanced=balanced,
    )


def _com_in_support(com_xy: XY, support: Sequence[XY]) -> bool:
    """True when the COM projects inside the contact hull.

    With fewer than three contacts there is no polygon to be inside, so we
    fall back to "is the COM between the contacts horizontally", which is the
    meaningful test for a two-point stance.
    """
    if len(support) >= 3:
        hull = convex_hull(support)
        if len(hull) >= 3:
            return point_in_poly(com_xy, hull)
    if len(support) == 2:
        lo, hi = sorted((support[0][0], support[1][0]))
        return lo <= com_xy[0] <= hi
    return False


def pull_direction(hand_xy: XY, shoulder_xy: XY, angle_deg: float) -> Tuple[float, float, float]:
    """Direction the climber pulls on a hand hold, in 3D wall coordinates.

    Returns a unit vector ``(u, v, n)`` where ``u``/``v`` are in the wall
    plane (matching board x/y) and ``n`` is out along the wall normal. On a
    vertical wall the pull is almost entirely in-plane and a hold's usable
    edge direction dominates; as the angle steepens the pull rotates out of
    the plane and holds become more about raw downward hanging, which is why
    edge orientation matters less on steep boards.
    """
    _, g_norm = gravity_components(angle_deg)

    du = hand_xy[0] - shoulder_xy[0]
    dv = hand_xy[1] - shoulder_xy[1]
    mag = math.hypot(du, dv)
    if mag < 1e-9:
        in_plane = (0.0, 1.0)
    else:
        in_plane = (du / mag, dv / mag)

    # The in-plane share shrinks as the off-wall pull grows.
    plane_share = math.sqrt(max(0.0, 1.0 - g_norm * g_norm))
    vec = (in_plane[0] * plane_share, in_plane[1] * plane_share, g_norm)
    norm = math.sqrt(sum(c * c for c in vec)) or 1.0
    return (vec[0] / norm, vec[1] / norm, vec[2] / norm)


def in_plane_pull_fraction(angle_deg: float) -> float:
    """Share of the pull vector that lies in the wall plane, in [0, 1].

    Used to fade the hold-orientation term out as the board steepens.
    """
    _, g_norm = gravity_components(angle_deg)
    return math.sqrt(max(0.0, 1.0 - g_norm * g_norm))


def foot_cut_threshold(angle_deg: float, base_reach_factor: float = 0.85) -> float:
    """Reach fraction past which a hand move is expected to cut a foot.

    Replaces the fixed `DYNAMIC_REACH_FACTOR` in `config.py`. On a vertical
    wall the feet stay on through almost any reach because they are bearing
    weight; on a steep board they are already near their tension limit, so a
    much smaller reach knocks them off.
    """
    g_par, _ = gravity_components(angle_deg)
    # Scales from ~1.0 of the base factor at vertical down to ~0.45 at roof.
    return base_reach_factor * (0.45 + 0.55 * g_par)


def angle_regime(angle_deg: float) -> str:
    """Coarse label for how the board behaves at this angle.

    Useful for explaining beta to a climber: the *reason* a beta changes with
    angle is that the wall crosses these boundaries.
    """
    probe = solve_contact_loads(
        hand_xys=[(0.0, 3.0 * COM_WALL_OFFSET)],
        foot_xys=[(0.0, 0.0)],
        com_xy=(0.0, 1.2 * COM_WALL_OFFSET),
        angle_deg=angle_deg,
    )
    if probe.tension_demand <= 0.0:
        return "slab"
    if probe.foot_engagement >= 0.5:
        return "tension"
    if probe.feet_secure:
        return "steep"
    return "roof"
