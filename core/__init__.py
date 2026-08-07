"""Angle-aware climbing core.

Replaces the angle-blind, heuristic-scored beam search with a model that
knows where gravity points relative to the wall. Layered as:

    params   -- calibration constants (fit these, don't tune them)
    physics  -- statics at a board angle: what holds you on, what pulls you off
    holds    -- per-hold quality and edge orientation
    state    -- body state as the primary object; per-limb tracks derive from it
    cost     -- what one limb transition costs at this angle, and why
    data     -- one training record per (climb, angle), no averaging

Search over these states lands next; the cost model is what makes it worth
searching.
"""

from .cost import CostModel
from .data import AngleClimb, iter_angle_climbs
from .holds import HoldInfo
from .physics import ContactLoads, angle_regime, gravity_components, solve_contact_loads
from .state import CUT, FLAGGING, ON_HOLD, Beta, BodyState, Contact, Move

__all__ = [
    "CostModel",
    "AngleClimb",
    "iter_angle_climbs",
    "HoldInfo",
    "ContactLoads",
    "solve_contact_loads",
    "gravity_components",
    "angle_regime",
    "BodyState",
    "Contact",
    "Move",
    "Beta",
    "ON_HOLD",
    "FLAGGING",
    "CUT",
]
