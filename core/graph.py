"""Angle-independent precompute for one hold set.

`ClimbGraph` captures everything about a set of holds that does *not* change
with board angle: positions, pairwise distances, which holds are physically
within reach of which others, and a canonical indexing scheme a later A*
search can key its state space on. Angle only ever affects the *cost* of an
edge in that search (via `core.cost.CostModel`), never whether the edge
exists at all -- so none of that belongs here. Building this once per hold
set and reusing it across every angle, every profile, and every beta is the
whole point: the reach/distance computation is the expensive part, and it is
identical for a 20 degree wall and a 70 degree one.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING

from .holds import HoldInfo

#: Kilter role ids (mirrors core.data / sequence_generator).
ROLE_START = 12
ROLE_HAND = 13
ROLE_FINISH = 14
ROLE_FOOT = 15
HAND_ROLES = (ROLE_START, ROLE_HAND, ROLE_FINISH)

#: y-distance within which two hand holds are considered "the same height"
#: for the purposes of sizing a coverage-bitmask window in the search. One
#: hold spacing is a reasonable band: holds set within a spacing of each
#: other on y are routinely used as alternates at the same point in a climb.
Y_TIE_BAND = 1.0 * X_SPACING


class ClimbGraph:
    """Angle-independent precompute for one hold set.

    Feet-vs-hands: `sequence_generator.is_valid_foot_transition` (the frozen
    legacy path) applies no role filter at all to its foot target -- it only
    checks occupancy and the foot reach matrix, which itself is built over
    *all* holds, not just role_id == 15. That confirms the Kilter convention
    referenced in the task: feet may use any placed hold, hand holds
    included, unless a climb explicitly restricts them. `feet_use_any_hold`
    defaults to `True` to match that behaviour.
    """

    def __init__(self, holds: Sequence[Dict], feet_use_any_hold: bool = True):
        # -- canonical order -------------------------------------------------
        # Sorted by hole_id (ties broken by placement_id, which is unique per
        # placement even if two holds somehow shared a hole_id). The exact
        # order doesn't matter physically -- what matters is that it is
        # stable and deterministic, since `hand_order` and every index array
        # below are defined relative to it, and a later search keys its
        # state space on positions within `hand_order`.
        canonical = sorted(
            holds, key=lambda h: (int(h["hole_id"]), int(h["placement_id"]))
        )
        self.holds: Tuple[Dict, ...] = tuple(canonical)
        n = len(self.holds)

        self.feet_use_any_hold = feet_use_any_hold

        self.xy = np.array(
            [(h["x"], h["y"]) for h in self.holds], dtype=float
        ).reshape(n, 2)
        self.hold_id = np.array([int(h["hole_id"]) for h in self.holds], dtype=int)
        self.placement_id = np.array(
            [int(h["placement_id"]) for h in self.holds], dtype=int
        )
        self.role = np.array([int(h["role_id"]) for h in self.holds], dtype=int)

        # -- role subsets ------------------------------------------------
        is_hand = np.isin(self.role, HAND_ROLES)
        self.hand_ix = np.nonzero(is_hand)[0]
        self.start_ix = np.nonzero(self.role == ROLE_START)[0]
        self.finish_ix = np.nonzero(self.role == ROLE_FINISH)[0]
        if feet_use_any_hold:
            self.foot_ix = np.arange(n, dtype=int)
        else:
            self.foot_ix = np.nonzero(self.role == ROLE_FOOT)[0]

        # -- hand_order / rank ---------------------------------------------
        # Sort hand_ix by (y, x, hold_id) ascending. y first because that is
        # the axis a climb progresses along; x and hold_id only break ties
        # deterministically.
        if self.hand_ix.size:
            hand_y = self.xy[self.hand_ix, 1]
            hand_x = self.xy[self.hand_ix, 0]
            hand_hid = self.hold_id[self.hand_ix]
            order_of_hand_ix = np.lexsort((hand_hid, hand_x, hand_y))
            self.hand_order = self.hand_ix[order_of_hand_ix]
        else:
            self.hand_order = np.array([], dtype=int)

        self.rank: Dict[int, int] = {
            int(self.hold_id[hold_idx]): pos
            for pos, hold_idx in enumerate(self.hand_order)
        }

        # -- distances / reach -----------------------------------------------
        # Broadcasted pairwise Euclidean distance -- O(N^2) but pure numpy,
        # no Python-level double loop.
        diff = self.xy[:, None, :] - self.xy[None, :, :]
        self.dist = np.sqrt((diff ** 2).sum(axis=-1))
        self.hand_reach = self.dist <= MAX_HAND_REACH
        self.foot_reach = self.dist <= MAX_FOOT_REACH

        self.hand_targets: List[np.ndarray] = _targets(self.hand_reach, self.hand_ix, n)
        self.foot_targets: List[np.ndarray] = _targets(self.foot_reach, self.foot_ix, n)

        # -- quality -----------------------------------------------------
        hold_info = HoldInfo()
        self.quality = np.array(
            [hold_info.quality(int(hid)) for hid in self.hold_id], dtype=float
        )

        # -- band --------------------------------------------------------
        # For each rank r (position within hand_order), the set of ranks
        # whose hand hold sits within Y_TIE_BAND of rank r's y. Includes r
        # itself -- this is a window of "roughly the same height", not a set
        # of alternates excluding the hold under consideration.
        m = self.hand_order.size
        if m:
            order_y = self.xy[self.hand_order, 1]
            y_diff = np.abs(order_y[:, None] - order_y[None, :])
            band_mask = y_diff <= Y_TIE_BAND
            self.band: List[np.ndarray] = [
                np.nonzero(band_mask[r])[0] for r in range(m)
            ]
        else:
            self.band = []

    # -- solvability checks --------------------------------------------------

    def connectivity_ok(self) -> bool:
        """Cheap necessary condition for solvability: every hand hold except
        the lowest-ranked one(s) must be within MAX_HAND_REACH of some other
        hand hold at or below it in y (so there's some way to reach it
        progressing upward). Returns False when a hand hold is provably
        unreachable from below -- this should catch most of what a later
        search would otherwise waste time discovering is unsolvable.
        """
        order = self.hand_order
        if order.size <= 1:
            return True

        ys = self.xy[order, 1]
        min_y = ys.min()
        for pos in range(order.size):
            y = ys[pos]
            if y <= min_y:
                # Lowest-ranked hold(s): nothing to reach up from.
                continue
            below_positions = np.nonzero(ys <= y)[0]
            below_positions = below_positions[below_positions != pos]
            if below_positions.size == 0:
                return False
            below_holds = order[below_positions]
            if not self.hand_reach[order[pos], below_holds].any():
                return False
        return True

    def max_y_gap(self) -> float:
        """Largest gap between consecutive ranks in hand_order, along y.

        Used to detect climbs likely to need the coverage-as-cost fallback
        (a later component; you're just exposing the number).
        """
        order = self.hand_order
        if order.size <= 1:
            return 0.0
        ys = self.xy[order, 1]
        diffs = np.diff(ys)
        return float(diffs.max())


def _targets(reach: np.ndarray, cols_ix: np.ndarray, n: int) -> List[np.ndarray]:
    """Per-row reachable subset of `cols_ix`, excluding the row's own index.

    `np.nonzero` on a boolean row returns indices in ascending order, so the
    result is already sorted -- satisfying the "sorted array" requirement
    without an extra sort.
    """
    col_mask = np.zeros(n, dtype=bool)
    col_mask[cols_ix] = True
    out: List[np.ndarray] = []
    for i in range(n):
        row = reach[i] & col_mask
        row = row.copy()
        row[i] = False
        out.append(np.nonzero(row)[0])
    return out
