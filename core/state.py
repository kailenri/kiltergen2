"""Body state as the primary object.

The legacy pipeline treats a climb as a flat list of ``{limb, hold}`` moves.
That representation cannot express *why* a move works, because the reason
lives in the body -- where the hips are, what is holding weight, what is
about to come off. Here the body state is primary and the per-limb move list
is a projection of it.

A state is hashable and immutable, so it can be a node in a graph search.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Tuple

from kinematics import Body, estimate_hip

from .params import COM_ABOVE_HIP
from .physics import ContactLoads, solve_contact_loads

XY = Tuple[float, float]

LIMBS: Tuple[str, ...] = ("RH", "LH", "RF", "LF")
HANDS: Tuple[str, ...] = ("RH", "LH")
FEET: Tuple[str, ...] = ("RF", "LF")

#: A limb is on a hold, flagging in space (no hold but still a stance), or
#: cut (contributing nothing).
ON_HOLD = "hold"
FLAGGING = "flag"
CUT = "cut"


@dataclass(frozen=True)
class Contact:
    """What one limb is doing."""

    kind: str = CUT
    hold_id: Optional[int] = None
    xy: Optional[XY] = None

    @property
    def is_weighted(self) -> bool:
        """True when this contact can carry load (a flag mostly cannot)."""
        return self.kind == ON_HOLD and self.xy is not None

    @property
    def is_stance(self) -> bool:
        """True when this contact contributes to balance, load-bearing or not."""
        return self.kind in (ON_HOLD, FLAGGING) and self.xy is not None


CUT_CONTACT = Contact()


@dataclass(frozen=True)
class BodyState:
    """An immutable snapshot of the whole body at one instant.

    `angle_deg` is part of the state because the same limb configuration is a
    different problem at 20 degrees than at 60 -- carrying it here means no
    downstream code can forget to account for it.
    """

    rh: Contact = CUT_CONTACT
    lh: Contact = CUT_CONTACT
    rf: Contact = CUT_CONTACT
    lf: Contact = CUT_CONTACT
    angle_deg: float = 40.0

    # -- access ------------------------------------------------------------

    @property
    def contacts(self) -> Dict[str, Contact]:
        return {"RH": self.rh, "LH": self.lh, "RF": self.rf, "LF": self.lf}

    def contact(self, limb: str) -> Contact:
        return getattr(self, limb.lower())

    def with_contact(self, limb: str, contact: Contact) -> "BodyState":
        return replace(self, **{limb.lower(): contact})

    def key(self) -> Tuple:
        """Hashable identity for a graph-search visited set.

        A hold-bearing contact is fully identified by `hold_id` -- the same
        hole always sits at the same xy, so two states on the same hold are
        the same state. A flagging contact has no `hold_id` (there is no
        hold), so its identity is carried entirely by `xy`. Earlier this
        method dropped `xy` for every contact, which meant two flag states
        with genuinely different foot positions -- and therefore different
        hips and different balance -- collapsed onto one key and one of them
        would be silently pruned from the search. `xy` is rounded rather
        than dropped: it is itself a deterministic function of the rest of
        the state (computed once by `compute_flag_position` from hip and the
        triggering hand target), so rounding only guards against float
        jitter -- it does not introduce new distinct states for situations
        that are physically the same.
        """
        def contact_key(c: Contact) -> Tuple:
            xy = (round(c.xy[0], 3), round(c.xy[1], 3)) if c.xy is not None else None
            return (c.kind, c.hold_id, xy)

        return tuple(
            (limb, contact_key(c)) for limb, c in sorted(self.contacts.items())
        )

    # -- derived geometry --------------------------------------------------

    def stance_points(self, limbs: Tuple[str, ...] = LIMBS) -> List[XY]:
        """Positions of contacts that contribute to balance."""
        return [self.contact(l).xy for l in limbs if self.contact(l).is_stance]

    def weighted_points(self, limbs: Tuple[str, ...] = LIMBS) -> List[XY]:
        """Positions of contacts that can carry load."""
        return [self.contact(l).xy for l in limbs if self.contact(l).is_weighted]

    def hip(self) -> Optional[XY]:
        """Hip estimate from the feet that are providing stance."""
        feet = self.stance_points(FEET)
        if feet:
            return estimate_hip(feet)
        # Feet fully cut: the body hangs beneath the hands.
        hands = self.weighted_points(HANDS)
        if not hands:
            return None
        anchor = estimate_hip(hands)
        return (anchor[0], anchor[1] - COM_ABOVE_HIP * 2.0)

    def com(self) -> Optional[XY]:
        """Centre of mass in the wall plane."""
        hip = self.hip()
        if hip is None:
            return None
        return (hip[0], hip[1] + COM_ABOVE_HIP)

    def loads(self, foot_quality: float = 1.0) -> Optional[ContactLoads]:
        """Resolve the forces holding this state at its board angle."""
        com = self.com()
        if com is None:
            return None
        return solve_contact_loads(
            hand_xys=self.weighted_points(HANDS),
            foot_xys=self.weighted_points(FEET),
            com_xy=com,
            angle_deg=self.angle_deg,
            foot_quality=foot_quality,
        )

    def to_body(self) -> Body:
        """Articulated pose, for the existing visualisation code."""
        limb_positions = {
            limb: (c.xy if c.kind == ON_HOLD else None)
            for limb, c in self.contacts.items()
        }
        cut = [l for l, c in self.contacts.items() if c.kind == CUT and l in FEET]
        flags = {
            l: (self.contact(l).xy if self.contact(l).kind == FLAGGING else None)
            for l in FEET
        }
        return Body.from_limb_positions(limb_positions, cut_feet=cut, flags=flags)

    @property
    def cut_feet(self) -> Tuple[str, ...]:
        return tuple(f for f in FEET if self.contact(f).kind == CUT)


@dataclass(frozen=True)
class Move:
    """One limb transition between two body states.

    `cost` and `loads` are attached at construction so the reason a move is
    hard travels with the move -- that is what makes a beta explainable
    rather than just a list of holds.
    """

    limb: str
    from_hold: Optional[int]
    to_hold: Optional[int]
    to_kind: str = ON_HOLD
    cost: float = 0.0
    #: Named cost contributions, for "this move is hard because...".
    breakdown: Dict[str, float] = None  # type: ignore[assignment]
    #: Side effects the move forced (feet cut, flag engaged).
    effects: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.breakdown is None:
            object.__setattr__(self, "breakdown", {})

    @property
    def is_hand(self) -> bool:
        return self.limb in HANDS

    def dominant_reason(self) -> Optional[str]:
        """The single largest cost contributor, for beta annotation."""
        if not self.breakdown:
            return None
        return max(self.breakdown.items(), key=lambda kv: kv[1])[0]


@dataclass
class Beta:
    """A complete solution: the state timeline plus its per-limb projection.

    Per-limb sequences are derived here rather than stored, so they cannot
    drift out of sync with the body states that justify them.
    """

    states: List[BodyState]
    moves: List[Move]
    angle_deg: float
    profile: str = "intermediate"
    #: True when this beta came from `core.search`'s coverage-as-cost
    #: fallback rather than the strict-order engine -- i.e. full hand-hold
    #: coverage was not required to reach this goal.
    relaxed: bool = False
    #: Hand-role hold ids the winning path never touched. Always empty
    #: unless `relaxed` is True.
    skipped_hold_ids: Tuple[int, ...] = ()
    #: `W_SKIP * len(skipped_hold_ids)` -- reported separately from
    #: `total_cost` rather than folded in, so `total_cost` always means
    #: "what the moves that were actually made cost".
    skip_penalty: float = 0.0

    @property
    def total_cost(self) -> float:
        return sum(m.cost for m in self.moves)

    @property
    def full_cost(self) -> float:
        """`total_cost` plus the skip penalty -- the number a caller should
        compare across betas when some may be relaxed and some may not.
        """
        return self.total_cost + self.skip_penalty

    def limb_track(self, limb: str) -> List[Tuple[int, Move]]:
        """This limb's own sequence: (timeline index, move) pairs.

        The index into `states` is kept so a limb's move can always be tied
        back to the whole-body position that made it possible.
        """
        return [(i, m) for i, m in enumerate(self.moves) if m.limb == limb]

    def limb_tracks(self) -> Dict[str, List[Tuple[int, Move]]]:
        return {limb: self.limb_track(limb) for limb in LIMBS}

    def crux(self) -> Optional[Tuple[int, Move]]:
        """The most expensive move -- the crux, by the cost model."""
        if not self.moves:
            return None
        idx = max(range(len(self.moves)), key=lambda i: self.moves[i].cost)
        return (idx, self.moves[idx])

    def __iter__(self) -> Iterator[Tuple[BodyState, Move]]:
        return iter(zip(self.states, self.moves))
