"""Tunable parameters for the angle-aware physics and cost model.

Kept separate from top-level `config.py`, which holds beam-search tuning for
the legacy generator. Everything here is a *calibration target*: the intent is
to fit these against per-angle community grades (`climb_stats`) rather than
hand-tune them. Defaults below are physically-motivated starting points.

Units are board units unless noted. `X_SPACING` = 18.667 board units = 8 in =
20.32 cm, so 1 board unit ~= 1.089 cm.
"""

from __future__ import annotations

from config import X_SPACING

# ---------------------------------------------------------------------------
# Body geometry (statics)
# ---------------------------------------------------------------------------

# How far the centre of mass sits out from the wall plane. A climber's hips
# ride roughly 20 cm off the surface, which is ~1.0 * X_SPACING. This is the
# single most important lever in the model: it sets the angle at which feet
# stop being weight-bearing (tan(theta) = COM_WALL_OFFSET / com_below_hands).
COM_WALL_OFFSET = 1.0 * X_SPACING

# COM sits above the hip by roughly this much (navel height relative to hip
# joint). Used when deriving COM from the articulated body pose.
COM_ABOVE_HIP = 1.2 * X_SPACING

# Floor on the hand-to-foot span used in the moment balance. Prevents a
# division blow-up when hands and feet are nearly level (e.g. a scrunched
# start position); below this the pose is treated as maximally compressed.
MIN_CONTACT_SPAN = 0.75 * X_SPACING

# ---------------------------------------------------------------------------
# Capacity limits
# ---------------------------------------------------------------------------

# Maximum inward moment the core/hip flexors can generate to hold the feet on
# when the statics say they would otherwise swing out, as a fraction of body
# weight. Beyond this the feet cut regardless of what the climber wants.
# 0.6 W puts the "feet blow off a static hang" boundary near 75 deg, which
# matches how Kilter angles above ~65 deg feel.
MAX_CORE_TENSION = 0.60

# Down-wall load a well-engaged foot can take, as a fraction of body weight.
# A foothold is an edge, not a friction plane: it supports down-wall force
# directly, so this is gated by engagement and hold quality, not by a
# Coulomb friction coefficient against the panel.
FOOT_EDGE_CAPACITY = 1.0

# Residual down-wall support from a foot that is *not* pressed into the wall
# (smearing, or hanging on a toe). Small but non-zero.
FOOT_SMEAR_RESIDUAL = 0.05

# ---------------------------------------------------------------------------
# Cost weights
# ---------------------------------------------------------------------------
# Each term is a fraction-of-bodyweight or a normalised [0, 1] quantity, so
# the weights are directly comparable. These are the primary regression
# targets for the calibration step.

W_GRIP = 1.00        # hand load / hold quality
W_TENSION = 0.80     # core demand to keep feet on
W_REACH = 0.60       # how close the move is to maximum reach
W_FOOT_CUT = 0.50    # penalty for losing a foot
W_HOLD_DIR = 0.40    # pulling against a hold's usable direction
W_BALANCE = 0.35     # COM outside the support polygon
W_CROSS = 0.25       # crossed / tangled limbs

# Flat per-move cost: every move costs time and tension regardless of how
# cheap its other terms are. Without this, W_GRIP's contribution -> 0 as the
# board approaches vertical (sin(0) = 0), so a shallow-angle search has
# nothing to price a move on and a bound built from these weights (the A*
# heuristic in search.py) degenerates to counting nothing per move.
W_MOVE = 0.15

# Penalty per hand hold left unused when the search falls back from strict
# coverage to soft coverage (traverses / down-climbs the prefix-order search
# can't reach). Large enough that skipping is always a last resort, not a
# cheap way to dodge an expensive hold.
W_SKIP = 4.0

# Reach cost ramps in only past this fraction of maximum reach; short moves
# are effectively free.
REACH_FREE_FRACTION = 0.55

# Neutral hold quality when the CV-derived orientation data has no entry.
DEFAULT_HOLD_QUALITY = 0.7

# ---------------------------------------------------------------------------
# Climber profile scaling
# ---------------------------------------------------------------------------
# A "beginner" beta and an "expert" beta are the same graph with different
# weights, not two systems. These multiply the weights above.

PROFILES = {
    # Minimise risk: avoid dynamic moves, big reaches, and foot cuts even at
    # the cost of extra moves.
    "beginner": {
        "W_REACH": 2.0,
        "W_FOOT_CUT": 2.5,
        "W_TENSION": 1.4,
        "W_BALANCE": 1.6,
    },
    # Neutral reference.
    "intermediate": {},
    # Minimise energy and move count; happy to cut feet and span.
    "expert": {
        "W_REACH": 0.6,
        "W_FOOT_CUT": 0.3,
        "W_TENSION": 0.8,
        "W_CROSS": 0.6,
    },
}
