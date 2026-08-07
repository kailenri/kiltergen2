"""Angle-conditioned cost of a single limb transition.

The cost of a move is dominated not by where the limb lands but by what the
*rest of the body* has to do while that limb is in the air. So every move is
evaluated against a transitional state -- the body with the moving limb
removed -- and the terms are read straight off the statics solve at the
board's angle.

Every term is a fraction of body weight or a normalised [0, 1] quantity, so
the weights in `params.py` are directly comparable and can be fit against
per-angle community grades instead of hand-tuned.

Scoring a move splits into two halves with very different call frequencies:

* `prepare(state, limb)` does everything that depends only on where the body
  *already* is -- the transitional statics solve, hand/foot quality, hip
  estimate -- and is the expensive half (the statics solve dominates).
* `score_prepared(ctx, target_xy, target_hold_id)` does the cheap,
  target-specific geometry (reach, hold direction, crossing) against the
  `MoveContext` `prepare` produced.

A search tries many candidate targets from the same `(state, limb)`, so
`prepare` is called once per limb-in-the-air and `score_prepared` once per
candidate hold -- instead of redoing the statics solve for every candidate,
which is what a single `score_move` call would do if it recomputed
everything from scratch each time. `score_move` is kept as a thin wrapper
over the two for callers that only need one-shot scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

from config import MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING

from . import params
from .holds import HoldInfo
from .physics import (
    ContactLoads,
    foot_cut_threshold,
    in_plane_pull_fraction,
    solve_contact_loads,
)
from .state import CUT, CUT_CONTACT, FEET, FLAGGING, HANDS, ON_HOLD, BodyState, Contact

XY = Tuple[float, float]

OPPOSITE_FOOT = {"RH": "LF", "LH": "RF"}

#: Every breakdown/feature key maps to the weight name that scales it. Kept
#: in one place so `breakdown[k] == weights[WEIGHT_NAME[k]] * features[k]`
#: is trivially true by construction rather than something each term has to
#: get right on its own.
WEIGHT_NAME: Dict[str, str] = {
    "grip": "W_GRIP",
    "tension": "W_TENSION",
    "reach": "W_REACH",
    "foot_cut": "W_FOOT_CUT",
    "hold_direction": "W_HOLD_DIR",
    "balance": "W_BALANCE",
    "cross": "W_CROSS",
    "move": "W_MOVE",
}


@dataclass(frozen=True)
class MoveContext:
    """Everything about scoring a move from `state` via `limb` that does NOT
    depend on the target -- computed once, reused across every candidate
    hold the search tries from this (state, limb).
    """

    state: BodyState
    limb: str
    transitional: BodyState
    loads: Optional[ContactLoads]
    hand_quality: float
    foot_quality: float
    n_hands: int
    hip: Optional[XY]
    shoulder: Optional[XY]
    #: grip / tension / balance / foot_cut / move -- target-independent
    #: terms, already weighted. A subset of these keys (foot_cut in
    #: particular) may still be topped up per-target in `score_prepared`.
    base_breakdown: Dict[str, float]
    #: Same terms, UNWEIGHTED (the raw physical quantity, not multiplied by
    #: the corresponding W_*).
    base_features: Dict[str, float]


class MoveScore(NamedTuple):
    total: float
    breakdown: Dict[str, float]
    effects: Tuple[str, ...]
    #: Unweighted per-term quantities, keyed exactly like `breakdown`, such
    #: that `breakdown[k] == weights[WEIGHT_NAME[k]] * features[k]` and
    #: `total == sum(weights[WEIGHT_NAME[k]] * v for k, v in features.items())`.
    features: Dict[str, float]


@dataclass
class CostModel:
    """Scores limb transitions at a fixed board angle.

    `profile` selects a weight set from `params.PROFILES`: the same graph
    scored for a beginner and for an expert yields genuinely different betas,
    which is the point -- not two separate systems.
    """

    angle_deg: float
    profile: str = "intermediate"
    holds: Optional[HoldInfo] = None

    def __post_init__(self):
        if self.holds is None:
            self.holds = HoldInfo()
        multipliers = params.PROFILES.get(self.profile, {})
        self.weights = {
            name: getattr(params, name) * multipliers.get(name, 1.0)
            for name in (
                "W_GRIP",
                "W_TENSION",
                "W_REACH",
                "W_FOOT_CUT",
                "W_HOLD_DIR",
                "W_BALANCE",
                "W_CROSS",
                "W_MOVE",
                # Not used by `prepare`/`score_prepared` -- every move a
                # search actually takes lands on a hold, never "skips" one.
                # Carried here anyway so `core.search`'s coverage-as-cost
                # fallback prices a skipped hold through the same
                # profile-scaled weight table as everything else, instead of
                # reaching past `CostModel` into `params.W_SKIP` directly.
                "W_SKIP",
            )
        }

    # -- public API --------------------------------------------------------

    def prepare(self, state: BodyState, limb: str) -> MoveContext:
        """Target-independent half of scoring a move: the statics solve.

        Everything here is a function of `(state, limb)` alone -- it is the
        same no matter which of the 8-13 candidate holds the search is about
        to try, so it is computed once and reused via `score_prepared`.
        """
        transitional = state.with_contact(limb, CUT_CONTACT)
        foot_quality = self._foot_quality(transitional)
        loads = transitional.loads(foot_quality=foot_quality)
        hand_quality = self._hand_quality(transitional)
        n_hands = max(1, len(transitional.weighted_points(HANDS)))

        hip = transitional.hip()
        shoulder = (hip[0], hip[1] + 2.0 * X_SPACING) if hip is not None else None

        base_breakdown: Dict[str, float] = {}
        base_features: Dict[str, float] = {}

        if loads is not None:
            per_hand = loads.hand_load / n_hands
            grip_feature = per_hand / max(0.15, hand_quality)
            base_features["grip"] = grip_feature
            base_breakdown["grip"] = self.weights["W_GRIP"] * grip_feature

            if loads.tension_demand > 0.0:
                base_features["tension"] = loads.tension_demand
                base_breakdown["tension"] = self.weights["W_TENSION"] * loads.tension_demand

            if not loads.balanced:
                base_features["balance"] = 1.0
                base_breakdown["balance"] = self.weights["W_BALANCE"]

            if not loads.feet_secure and transitional.weighted_points(FEET):
                base_features["foot_cut"] = 1.0
                base_breakdown["foot_cut"] = self.weights["W_FOOT_CUT"]

        # Flat per-move charge -- always present, never scaled by anything.
        base_features["move"] = 1.0
        base_breakdown["move"] = self.weights["W_MOVE"]

        return MoveContext(
            state=state,
            limb=limb,
            transitional=transitional,
            loads=loads,
            hand_quality=hand_quality,
            foot_quality=foot_quality,
            n_hands=n_hands,
            hip=hip,
            shoulder=shoulder,
            base_breakdown=base_breakdown,
            base_features=base_features,
        )

    def score_prepared(
        self,
        ctx: MoveContext,
        target_xy: XY,
        target_hold_id: Optional[int],
    ) -> MoveScore:
        """Target-dependent half of scoring a move: reach, direction, cross."""
        limb = ctx.limb
        breakdown: Dict[str, float] = dict(ctx.base_breakdown)
        features: Dict[str, float] = dict(ctx.base_features)
        effects: list = []

        if "foot_cut" in breakdown:
            effects.append("feet_blow_off")

        # -- how far the limb has to travel --------------------------------
        reach_cost, reach_fraction, reach_feature = self._reach_cost(
            ctx.state, limb, target_xy
        )
        if reach_cost > 0.0:
            breakdown["reach"] = reach_cost
            features["reach"] = reach_feature

        # A long hand reach at a steep angle strips a foot even when the
        # static pose would have held. The threshold itself is angle-derived.
        if limb in HANDS and reach_fraction > foot_cut_threshold(self.angle_deg):
            if "foot_cut" not in breakdown:
                breakdown["foot_cut"] = self.weights["W_FOOT_CUT"]
                features["foot_cut"] = 1.0
                effects.append("dynamic_cut")

        # -- pulling against the hold's usable direction --------------------
        if limb in HANDS:
            dir_cost, dir_feature = self._hold_direction_cost(
                ctx, target_xy, target_hold_id
            )
            if dir_cost > 0.0:
                breakdown["hold_direction"] = dir_cost
                features["hold_direction"] = dir_feature

        # -- crossed limbs --------------------------------------------------
        cross_cost, cross_feature = self._cross_cost(ctx, target_xy)
        if cross_cost > 0.0:
            breakdown["cross"] = cross_cost
            features["cross"] = cross_feature
            effects.append("cross")

        return MoveScore(
            total=sum(breakdown.values()),
            breakdown=breakdown,
            effects=tuple(effects),
            features=features,
        )

    def score_move(
        self,
        state: BodyState,
        limb: str,
        target_xy: XY,
        target_hold_id: Optional[int],
    ) -> Tuple[float, Dict[str, float], Tuple[str, ...]]:
        """Cost of moving `limb` to `target_xy`.

        Returns ``(total, breakdown, effects)``. `breakdown` names each
        contribution so the beta can say *why* a move is hard; `effects`
        records what the move forced, such as a foot cutting.

        Unchanged signature/behavior -- implemented as `prepare()` +
        `score_prepared()`. Callers scoring many targets from the same
        `(state, limb)` should call those directly instead.
        """
        ctx = self.prepare(state, limb)
        s = self.score_prepared(ctx, target_xy, target_hold_id)
        return (s.total, s.breakdown, s.effects)

    def score_state(self, state: BodyState) -> Optional[Dict[str, float]]:
        """Standing cost of holding a position, independent of any move.

        Used to rank rest positions and to explain why a stance is or is not
        somewhere a climber can recover.
        """
        loads = state.loads(foot_quality=self._foot_quality(state))
        if loads is None:
            return None
        n_hands = max(1, len(state.weighted_points(HANDS)))
        return {
            "hand_load": loads.hand_load,
            "per_hand": loads.hand_load / n_hands,
            "tension": loads.tension_demand,
            "foot_engagement": loads.foot_engagement,
            "balanced": float(loads.balanced),
            "secure": float(loads.feet_secure),
        }

    # -- terms -------------------------------------------------------------

    def _reach_cost(
        self, state: BodyState, limb: str, target_xy: XY
    ) -> Tuple[float, float, float]:
        """Cost of the span itself, the raw fraction-of-max-reach, and the
        unweighted reach feature (0.0 when inside the free zone)."""
        current = state.contact(limb)
        max_reach = MAX_HAND_REACH if limb in HANDS else MAX_FOOT_REACH
        if current.xy is None:
            return (0.0, 0.0, 0.0)
        dist = math.hypot(target_xy[0] - current.xy[0], target_xy[1] - current.xy[1])
        fraction = dist / max_reach if max_reach > 0 else 0.0
        if fraction <= params.REACH_FREE_FRACTION:
            return (0.0, fraction, 0.0)
        # Ramp quadratically past the free zone: the last inch of reach costs
        # far more than the first.
        over = (fraction - params.REACH_FREE_FRACTION) / (
            1.0 - params.REACH_FREE_FRACTION
        )
        feature = over * over
        return (self.weights["W_REACH"] * feature, fraction, feature)

    def _hold_direction_cost(
        self,
        ctx: MoveContext,
        target_xy: XY,
        target_hold_id: Optional[int],
    ) -> Tuple[float, float]:
        """Penalty for pulling across a hold's usable edge.

        Faded out as the board steepens: on a roof the pull is almost normal
        to the wall, so which way an edge faces stops mattering much.

        Uses the hip/shoulder estimate from the transitional state (the
        moving hand already lifted) rather than re-deriving it with the hand
        landed at `target_xy`. The two agree whenever feet are providing
        stance, which is the near-universal case -- they can only differ
        when both feet are already cut and the hip estimate falls back to
        hanging beneath the hands, in which case using the pre-landing
        estimate keeps this term target-independent and reusable across
        candidates, at the cost of a same-order approximation in a rare
        situation.
        """
        if ctx.hip is None:
            return (0.0, 0.0)
        shoulder = ctx.shoulder
        pull_u = target_xy[0] - shoulder[0]
        pull_v = target_xy[1] - shoulder[1]

        alignment = self.holds.pull_alignment(target_hold_id, pull_u, pull_v)
        if alignment is None:
            return (0.0, 0.0)
        plane_share = in_plane_pull_fraction(self.angle_deg)
        feature = (1.0 - alignment) * plane_share
        return (self.weights["W_HOLD_DIR"] * feature, feature)

    def _cross_cost(self, ctx: MoveContext, target_xy: XY) -> Tuple[float, float]:
        """Penalty for limbs ending up on the wrong side of each other."""
        limb = ctx.limb
        if limb in HANDS:
            other = "LH" if limb == "RH" else "RH"
        else:
            other = "LF" if limb == "RF" else "RF"
        other_xy = ctx.state.contact(other).xy
        if other_xy is None:
            return (0.0, 0.0)
        pad = 0.5 * X_SPACING
        crossed = (
            target_xy[0] < other_xy[0] - pad
            if limb in ("RH", "RF")
            else target_xy[0] > other_xy[0] + pad
        )
        if not crossed:
            return (0.0, 0.0)
        # A cross is cheaper when the feet are solidly on to twist against.
        # This is genuinely target-dependent (landing the moving limb can
        # shift the hip/COM enough to change foot engagement), so it is
        # resolved here rather than in `prepare`.
        landed = ctx.state.with_contact(
            limb, Contact(kind=ON_HOLD, hold_id=None, xy=target_xy)
        )
        loads = landed.loads()
        relief = loads.foot_engagement if loads is not None else 0.0
        feature = 1.0 - 0.5 * relief
        return (self.weights["W_CROSS"] * feature, feature)

    # -- helpers -----------------------------------------------------------

    def _hand_quality(self, state: BodyState) -> float:
        ids = [
            state.contact(h).hold_id
            for h in HANDS
            if state.contact(h).is_weighted
        ]
        if not ids:
            return params.DEFAULT_HOLD_QUALITY
        return sum(self.holds.quality(i) for i in ids) / len(ids)

    def _foot_quality(self, state: BodyState) -> float:
        ids = [
            state.contact(f).hold_id
            for f in FEET
            if state.contact(f).is_weighted
        ]
        if not ids:
            return params.DEFAULT_HOLD_QUALITY
        return sum(self.holds.quality(i) for i in ids) / len(ids)


def resolve_move_effects(
    state: BodyState, limb: str, target_xy: XY, effects: Tuple[str, ...]
) -> BodyState:
    """Apply a move's physical consequences to produce the next state.

    Keeps foot-cut and flag bookkeeping in one place so the search and any
    replay of a stored beta cannot disagree about what the body did.
    """
    nxt = state
    if "feet_blow_off" in effects or "dynamic_cut" in effects:
        preferred = OPPOSITE_FOOT.get(limb)
        candidates = [preferred] if preferred else []
        candidates += [f for f in FEET if f != preferred]
        for foot in candidates:
            if foot and nxt.contact(foot).is_weighted:
                nxt = nxt.with_contact(foot, CUT_CONTACT)
                break
    return nxt
