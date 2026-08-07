"""Weighted A* over `BodyState` -- turns a `ClimbGraph` + `CostModel` into a
`Beta`.

State space
-----------
A node is a `SearchState`: the full `BodyState` (all four limbs) plus a
compact encoding of *which hand-role holds have been touched by a hand at
some point in the search so far* -- `(base, mask)`. Ranks `[0, base)` are all
covered; `mask` is a bitmask over the next `W` ranks `[base, base+W)`, with
bit 0 always clear (canonicalized after every update so `(base, mask)` is a
unique, minimal encoding of the covered-rank set). Feet carry no progress
coordinate: they can move freely without affecting `(base, mask)`.

This keeps the state space small enough for A* to explore -- coverage is
tracked as a rolling window over the hand-hold order rather than a full
`2**N` bitmask over every hand hold in the climb -- while still capturing
Kilter's real completion rule (every hand-role hold visited at some point,
not just currently held).

Heuristic
---------
`h(s) = heuristic_weight * remaining_hand_holds * floor`, where `floor` is a
provable per-hand-move lower bound (`move_cost_floor`): every uncovered hand
hold needs its own hand move, distinct holds need distinct moves, and every
hand move costs at least `floor` (see `move_cost_floor`'s docstring for the
proof, which is checked against `core/cost.py`'s actual formulas by
`test_core_search.py`'s admissibility test rather than merely asserted).
`heuristic_weight=1.0` is admissible; `heuristic_weight=2.2` is the fast
default and no longer guaranteed admissible, which is why it is a parameter
and not baked in.

Production entry point
-----------------------
`solve` is this module's proven-correct core (admissibility, replay
consistency, bounds) but at its literal defaults only converges on ~21-33%
of real climbs -- see `test_core_search.py`'s module docstring for the
measured numbers and root cause (the cost model's `tension`/`balance`/
`foot_cut` terms, not a search bug). `solve_or_relax` is what callers
outside this module's own tests should reach for: it retries a failed fast
solve with a larger budget (fixes the majority-share search-budget gap
without slowing down the common case that already succeeds), then falls
back to `_solve_relaxed`'s coverage-as-cost engine for the minority of
climbs (real traverses/down-climbs) the strict-order engine can't solve at
any budget. `solve_k` builds on `solve_or_relax` to return several
diverse betas instead of one.
"""

from __future__ import annotations

import heapq
import itertools
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from config import MAX_FOOT_REACH
from kinematics import choose_flag_foot, compute_flag_position

from . import params
from .cost import CostModel, resolve_move_effects
from .graph import ClimbGraph
from .physics import gravity_components
from .state import (
    CUT_CONTACT,
    FEET,
    FLAGGING,
    HANDS,
    LIMBS,
    ON_HOLD,
    Beta,
    BodyState,
    Contact,
    Move,
)

XY = Tuple[float, float]

#: (body_key, base, mask) -- the closed/visited-set identity for a node.
NodeKey = Tuple[tuple, int, int]

#: Cap on how many candidate targets a single limb considers per expansion,
#: taking the nearest-by-distance when a reach-legal candidate set is
#: larger. A climber overwhelmingly uses the nearest usable hold among the
#: ones in reach; a hold twice as far almost always loses on the reach term
#: anyway (`CostModel._reach_cost` ramps quadratically past
#: `REACH_FREE_FRACTION`), so this keeps per-node branching bounded on
#: densely-set hold clusters without changing which candidate ends up
#: cheapest in practice. Every candidate evaluated still goes through the
#: full `CostModel`, so this only prunes which targets get considered, not
#: how any one of them is scored.
MAX_HAND_CANDIDATES = 8
MAX_FOOT_CANDIDATES = 6


class SearchState(NamedTuple):
    body: BodyState
    base: int   # ranks [0, base) of graph.hand_order are all covered
    mask: int   # bitmask over ranks [base, base+W) -- bit 0 always clear


# ---------------------------------------------------------------------------
# Coverage bookkeeping
# ---------------------------------------------------------------------------


def is_legal_rank(rank: Optional[int], base: int, mask: int, W: int) -> bool:
    """True when landing a hand on this rank is legal under strict coverage.

    `rank is None` means the target has no coverage effect at all (not a
    hand-role hold) -- always legal with respect to coverage. Otherwise legal
    iff it re-covers an already-covered rank (`rank < base`) or advances
    within the open window (`base <= rank < base + W`).
    """
    if rank is None:
        return True
    if rank < base:
        return True
    return rank < base + W


def cover_rank(base: int, mask: int, rank: Optional[int]) -> Tuple[int, int]:
    """Mark `rank` covered and canonicalize `(base, mask)`.

    Re-covering an already-covered rank (`rank < base`) is a no-op. Setting a
    bit past `base` and then rolling `base` forward while bit 0 is set keeps
    `(base, mask)` the unique, minimal encoding of the covered-rank set --
    the same canonicalization is used whether this is called during seeding
    or during expansion, so the two never disagree about what "covered"
    means.
    """
    if rank is None or rank < base:
        return base, mask
    bit = rank - base
    mask = mask | (1 << bit)
    while mask & 1:
        base += 1
        mask >>= 1
    return base, mask


def popcount_covered(base: int, mask: int) -> int:
    return base + bin(mask).count("1")


def _adaptive_window(graph: ClimbGraph) -> int:
    """W = clamp(largest y-tie band size in the climb, 1, 4)."""
    if not graph.band:
        return 1
    w = max(len(b) for b in graph.band)
    return max(1, min(4, w))


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


def move_cost_floor(angle_deg: float, weights: Dict[str, float], best_hand_quality: float) -> float:
    """Provable per-hand-move lower bound.

    Every scored move pays the flat `W_MOVE` charge unconditionally
    (`CostModel.prepare` always adds it, regardless of anything else -- see
    `core/cost.py`). Every *hand* move that reaches this floor's caller
    (`h`) also always has a `grip` term, because in every state this search
    can reach, both hands are permanently `ON_HOLD` (seeds place them there
    and no move in `_expand`/`resolve_move_effects` ever cuts a hand) --
    so the transitional state used by `CostModel.prepare` (the state with
    only the *moving* hand removed) always still has the *other* hand
    weighted, which makes `state.loads()` non-`None` and puts `grip` in the
    breakdown unconditionally.

    Given that, `grip`'s weighted contribution is
    `W_GRIP * (hand_load / n_hands) / max(0.15, hand_quality)`, and:

    * `n_hands == 1` for a hand move (the other hand is the only one left
      weighted in the transitional state), so `hand_load / n_hands ==
      hand_load`.
    * `hand_load = hypot(hand_normal, hand_shear) >= hand_normal =
      g_norm + foot_normal >= g_norm = sin(angle)` (`foot_normal >= 0`
      always in `solve_contact_loads`) -- so `hand_load` can never fall
      below `sin(theta)` regardless of how well the feet are engaged. This
      is the "best case is a fully engaged stance" claim.
    * `hand_quality` is the quality of whichever hand-role hold the
      surviving hand is actually on, which is at most `best_hand_quality`
      (the best hand-role hold quality anywhere in the graph) by
      definition of "best" -- so dividing by the graph-wide best can only
      make this floor's grip term smaller than or equal to the real one.

    Every other breakdown term (`tension`, `reach`, `foot_cut`,
    `hold_direction`, `balance`, `cross`) is either absent or >= 0 by
    construction (each is `weight * feature` with `weight > 0` and
    `feature >= 0`), so they can only add to the total, never subtract from
    it. Hence every reachable hand move costs >= `W_MOVE + W_GRIP *
    sin(angle) / max(0.15, best_hand_quality)`, exactly the value returned
    here.
    """
    _, g_norm = gravity_components(angle_deg)
    return weights["W_MOVE"] + weights["W_GRIP"] * g_norm / max(0.15, best_hand_quality)


def h(search_state: SearchState, graph: ClimbGraph, floor: float, heuristic_weight: float) -> float:
    remaining = graph.hand_order.size - popcount_covered(search_state.base, search_state.mask)
    return heuristic_weight * remaining * floor


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def _rank_foot_candidates(
    graph: ClimbGraph, hand_idx: int, mid_x: float, prefer_right: bool, skip_idxs: set
) -> List[Tuple[float, int]]:
    """Legacy `_select_initial_feet` ranking, adapted to `ClimbGraph` arrays.

    `cost = dist + 2*max(0, foot_y - hand_y) - (5 if same_side else 0)`,
    restricted to `MAX_FOOT_REACH`. Returns `(cost, foot_index)` pairs,
    ascending -- cheapest (best) first.
    """
    hand_y = float(graph.xy[hand_idx, 1])
    out: List[Tuple[float, int]] = []
    for fi in graph.foot_ix.tolist():
        if fi in skip_idxs:
            continue
        d = float(graph.dist[hand_idx, fi])
        if d > MAX_FOOT_REACH:
            continue
        fx = float(graph.xy[fi, 0])
        fy = float(graph.xy[fi, 1])
        side_match = (fx >= mid_x) if prefer_right else (fx <= mid_x)
        y_above = max(0.0, fy - hand_y)
        cost = d + 2.0 * y_above - (5.0 if side_match else 0.0)
        out.append((cost, fi))
    out.sort(key=lambda t: t[0])
    return out


def seed_states(
    graph: ClimbGraph, cost: CostModel, angle_deg: float, *, n_foot_seeds: int = 5
) -> List[Tuple[SearchState, float]]:
    """Enumerate starting `SearchState`s. All seeds carry `g = 0.0` -- the
    seed IS the starting position, not a move into it.
    """
    seeds: List[Tuple[SearchState, float]] = []
    starts = graph.start_ix
    if starts.size == 0:
        return []

    if starts.size <= 1:
        s = int(starts[0])
        pairs = [(s, s)]
    else:
        pairs = [(int(r), int(l)) for r in starts.tolist() for l in starts.tolist() if r != l]

    for rh_idx, lh_idx in pairs:
        skip = {rh_idx, lh_idx}
        mid_x = (float(graph.xy[rh_idx, 0]) + float(graph.xy[lh_idx, 0])) / 2.0
        rf_ranked = _rank_foot_candidates(graph, rh_idx, mid_x, True, skip)
        lf_ranked = _rank_foot_candidates(graph, lh_idx, mid_x, False, skip)

        rh_hold_id = int(graph.hold_id[rh_idx])
        lh_hold_id = int(graph.hold_id[lh_idx])
        rh_xy: XY = (float(graph.xy[rh_idx, 0]), float(graph.xy[rh_idx, 1]))
        lh_xy: XY = (float(graph.xy[lh_idx, 0]), float(graph.xy[lh_idx, 1]))

        base0, mask0 = cover_rank(0, 0, graph.rank.get(rh_hold_id))
        base0, mask0 = cover_rank(base0, mask0, graph.rank.get(lh_hold_id))

        def make_body(rf_idx: Optional[int], lf_idx: Optional[int]) -> BodyState:
            rf_contact = CUT_CONTACT
            lf_contact = CUT_CONTACT
            if rf_idx is not None:
                rf_contact = Contact(
                    kind=ON_HOLD,
                    hold_id=int(graph.hold_id[rf_idx]),
                    xy=(float(graph.xy[rf_idx, 0]), float(graph.xy[rf_idx, 1])),
                )
            if lf_idx is not None:
                lf_contact = Contact(
                    kind=ON_HOLD,
                    hold_id=int(graph.hold_id[lf_idx]),
                    xy=(float(graph.xy[lf_idx, 0]), float(graph.xy[lf_idx, 1])),
                )
            return BodyState(
                rh=Contact(kind=ON_HOLD, hold_id=rh_hold_id, xy=rh_xy),
                lh=Contact(kind=ON_HOLD, hold_id=lh_hold_id, xy=lh_xy),
                rf=rf_contact,
                lf=lf_contact,
                angle_deg=angle_deg,
            )

        if rf_ranked and lf_ranked:
            buffer = n_foot_seeds + 2
            combos = []
            for rc, rf in rf_ranked[:buffer]:
                for lc, lf in lf_ranked[:buffer]:
                    if rf == lf:
                        continue
                    combos.append((rc + lc, rf, lf))
            combos.sort(key=lambda t: t[0])
            seen_pairs: set = set()
            n_added = 0
            for _, rf, lf in combos:
                if (rf, lf) in seen_pairs:
                    continue
                seen_pairs.add((rf, lf))
                body = make_body(rf, lf)
                seeds.append((SearchState(body, base0, mask0), 0.0))
                n_added += 1
                if n_added >= n_foot_seeds:
                    break

        # Feet-cut hanging start -- always seeded (roof-angle climbs).
        body = make_body(None, None)
        seeds.append((SearchState(body, base0, mask0), 0.0))

    return seeds


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def _foot_targets_from_cut(graph: ClimbGraph, state: BodyState) -> np.ndarray:
    """Candidate landing holds for a foot that is currently off the wall
    (CUT or FLAGGING): any foot-usable hold within `MAX_FOOT_REACH` of the
    current hip estimate, or every foot-usable hold if there is no hip
    estimate at all.
    """
    hip = state.hip()
    if hip is None:
        return graph.foot_ix
    dx = graph.xy[graph.foot_ix, 0] - hip[0]
    dy = graph.xy[graph.foot_ix, 1] - hip[1]
    within = np.hypot(dx, dy) <= MAX_FOOT_REACH
    return graph.foot_ix[within]


def _cap_nearest(idxs: np.ndarray, from_xy: XY, graph: ClimbGraph, k: int) -> np.ndarray:
    """Keep only the `k` nearest-by-distance entries of `idxs` to `from_xy`.

    A no-op when `idxs` is already at or under the cap. See `MAX_HAND_CANDIDATES`
    / `MAX_FOOT_CANDIDATES` for why distance is the right, cheap-to-compute
    proxy here.
    """
    if len(idxs) <= k:
        return idxs
    dx = graph.xy[idxs, 0] - from_xy[0]
    dy = graph.xy[idxs, 1] - from_xy[1]
    d = np.hypot(dx, dy)
    order = np.argsort(d, kind="stable")[:k]
    return idxs[order]


def _fast_with_contact(state: BodyState, limb: str, contact: Contact) -> BodyState:
    """Equivalent to `state.with_contact(limb, contact)`.

    `BodyState.with_contact` goes through `dataclasses.replace`, which
    re-derives the field list and does a `getattr` per untouched field on
    every call -- measurable overhead at the volume `_expand` calls it
    (thousands of times per solve). Building the replacement directly
    through `BodyState`'s ordinary public constructor is behaviourally
    identical (same immutable dataclass, same fields) and sidesteps that
    reflection; `core/state.py` itself is untouched.
    """
    if limb == "RH":
        return BodyState(rh=contact, lh=state.lh, rf=state.rf, lf=state.lf, angle_deg=state.angle_deg)
    if limb == "LH":
        return BodyState(rh=state.rh, lh=contact, rf=state.rf, lf=state.lf, angle_deg=state.angle_deg)
    if limb == "RF":
        return BodyState(rh=state.rh, lh=state.lh, rf=contact, lf=state.lf, angle_deg=state.angle_deg)
    return BodyState(rh=state.rh, lh=state.lh, rf=state.rf, lf=contact, angle_deg=state.angle_deg)


def _expand(
    ss: SearchState,
    graph: ClimbGraph,
    cost: CostModel,
    idx_of_hold_id: Dict[int, int],
    W: int,
    *,
    hand_hold_penalty: Optional[Dict[int, float]] = None,
    relaxed: bool = False,
):
    """Yield `(next_search_state, move, search_cost)` for every legal
    limb transition out of `ss`.

    `search_cost` is what the caller should add to `g` -- it is `move.cost`
    (the real, physical, unweighted-features-preserving cost from
    `CostModel`) plus any `hand_hold_penalty` for the target, which exists
    only to steer `solve_k`'s re-search toward a different hand sequence and
    is deliberately kept out of `move.cost`/`Beta.total_cost` so those still
    mean exactly `w . Phi` (see `core/cost.py` module docstring).

    `relaxed=True` drops both halves of the strict-order coverage machinery:
    hand targets are legal in any order (no `is_legal_rank` filtering), and
    `(base, mask)` is passed through unchanged rather than advanced by
    `cover_rank`. This is `core.search`'s coverage-as-cost fallback --
    coverage is priced after the fact by `_solve_relaxed`, not gated here.
    """
    state = ss.body
    base, mask = ss.base, ss.mask

    for limb in LIMBS:
        contact = state.contact(limb)

        if limb in HANDS:
            cur_idx = idx_of_hold_id.get(contact.hold_id)
            if cur_idx is None:
                continue
            cur_xy: XY = (float(graph.xy[cur_idx, 0]), float(graph.xy[cur_idx, 1]))
            if relaxed:
                legal = list(graph.hand_targets[cur_idx])
            else:
                # Coverage legality is filtered *before* the nearest-K cap,
                # so a farther-but-legal target is never squeezed out by a
                # nearer but already-covered one.
                legal = [
                    t
                    for t in graph.hand_targets[cur_idx]
                    if is_legal_rank(graph.rank.get(int(graph.hold_id[t])), base, mask, W)
                ]
            target_ixs = (
                _cap_nearest(np.array(legal, dtype=int), cur_xy, graph, MAX_HAND_CANDIDATES)
                if legal
                else np.array([], dtype=int)
            )
        else:
            if contact.kind == ON_HOLD:
                cur_idx = idx_of_hold_id.get(contact.hold_id)
                if cur_idx is None:
                    continue
                cur_xy = (float(graph.xy[cur_idx, 0]), float(graph.xy[cur_idx, 1]))
                target_ixs = _cap_nearest(
                    graph.foot_targets[cur_idx], cur_xy, graph, MAX_FOOT_CANDIDATES
                )
            else:
                raw = _foot_targets_from_cut(graph, state)
                hip = state.hip()
                if hip is not None:
                    target_ixs = _cap_nearest(raw, hip, graph, MAX_FOOT_CANDIDATES)
                else:
                    target_ixs = raw[: MAX_FOOT_CANDIDATES]

        if len(target_ixs) == 0:
            continue

        ctx = cost.prepare(state, limb)

        for t in target_ixs:
            t = int(t)
            target_hold_id = int(graph.hold_id[t])
            target_xy: XY = (float(graph.xy[t, 0]), float(graph.xy[t, 1]))
            rank = graph.rank.get(target_hold_id) if limb in HANDS else None

            score = cost.score_prepared(ctx, target_xy, target_hold_id)

            landed = _fast_with_contact(
                state, limb, Contact(kind=ON_HOLD, hold_id=target_hold_id, xy=target_xy)
            )
            next_body = resolve_move_effects(landed, limb, target_xy, score.effects)
            effects = score.effects

            # Flag handling: a hand move that crossed but did not also cut a
            # foot may flag the opposite foot instead of leaving it planted
            # unexplained. Cut wins -- only checked when neither cut effect
            # fired.
            if (
                limb in HANDS
                and "cross" in effects
                and "feet_blow_off" not in effects
                and "dynamic_cut" not in effects
            ):
                foot_positions = {
                    f: (state.contact(f).xy if state.contact(f).is_weighted else None)
                    for f in FEET
                }
                flag_foot = choose_flag_foot(limb, foot_positions)
                if flag_foot is not None:
                    hip = state.hip()  # pre-move hip -- the foot hasn't lifted yet
                    if hip is not None:
                        flag_xy = compute_flag_position(hip, target_xy, flag_foot)
                        next_body = _fast_with_contact(
                            next_body, flag_foot, Contact(kind=FLAGGING, hold_id=None, xy=flag_xy)
                        )
                        effects = effects + ("flagged",)

            if limb in HANDS and not relaxed:
                new_base, new_mask = cover_rank(base, mask, rank)
            else:
                new_base, new_mask = base, mask

            move = Move(
                limb=limb,
                from_hold=contact.hold_id,
                to_hold=target_hold_id,
                to_kind=ON_HOLD,
                cost=score.total,
                breakdown=score.breakdown,
                effects=effects,
            )
            search_cost = score.total
            if limb in HANDS and hand_hold_penalty:
                search_cost += hand_hold_penalty.get(target_hold_id, 0.0)
            yield SearchState(next_body, new_base, new_mask), move, search_cost


# ---------------------------------------------------------------------------
# Goal test
# ---------------------------------------------------------------------------


def is_goal(
    graph: ClimbGraph,
    ss: SearchState,
    *,
    matched_finish: bool = True,
    require_foot_on: bool = True,
    require_coverage: bool = True,
) -> bool:
    if require_coverage and popcount_covered(ss.base, ss.mask) != graph.hand_order.size:
        return False

    finish_ix = graph.finish_ix
    if finish_ix.size == 0:
        return False

    state = ss.body
    finish_ids = {int(graph.hold_id[i]) for i in finish_ix}
    hand_ids = {state.rh.hold_id, state.lh.hold_id}
    hand_ids.discard(None)

    if finish_ix.size == 1:
        fid = next(iter(finish_ids))
        if matched_finish:
            ok = state.rh.hold_id == fid and state.lh.hold_id == fid
        else:
            ok = fid in hand_ids
    else:
        ok = finish_ids.issubset(hand_ids)

    if not ok:
        return False

    if require_foot_on and not (state.rf.is_weighted or state.lf.is_weighted):
        return False

    return True


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

#: Raw path pieces `_astar` hands back: the state timeline and the moves
#: between them, before a caller decides whether/how to wrap them in a
#: `Beta` (`solve` always does; `_solve_relaxed` adds skip bookkeeping first).
PathResult = Tuple[List[BodyState], List[Move]]


def _astar(
    graph: ClimbGraph,
    cost: CostModel,
    *,
    heuristic_weight: float,
    window: Optional[int],
    max_expansions: int,
    time_budget_s: float,
    matched_finish: bool,
    require_foot_on: bool,
    n_foot_seeds: int,
    hand_hold_penalty: Optional[Dict[int, float]] = None,
    relaxed: bool = False,
) -> Optional[PathResult]:
    """The weighted A* loop shared by `solve` and `_solve_relaxed`.

    `relaxed=True` drops strict-order coverage on both ends: `_expand` stops
    filtering/advancing `(base, mask)` (see its docstring), the goal test
    stops requiring full coverage, and the heuristic -- which is defined in
    terms of *remaining covered ranks*, a quantity that no longer moves when
    coverage isn't tracked -- is disabled (`heuristic_weight` is overridden
    to 0.0) rather than left plugged in and silently useless. `h ≡ 0` still
    makes this correct, plain Dijkstra over a smaller state space (dedup is
    on `BodyState.key()` alone, since `(base, mask)` no longer varies).
    """
    eff_heuristic_weight = 0.0 if relaxed else heuristic_weight
    require_coverage = not relaxed

    W = window if window is not None else _adaptive_window(graph)

    seeds = seed_states(graph, cost, cost.angle_deg, n_foot_seeds=n_foot_seeds)
    if not seeds:
        return None

    idx_of_hold_id: Dict[int, int] = {int(h): i for i, h in enumerate(graph.hold_id)}

    if graph.hand_ix.size:
        best_hand_quality = float(graph.quality[graph.hand_ix].max())
    else:
        best_hand_quality = params.DEFAULT_HOLD_QUALITY
    floor = move_cost_floor(cost.angle_deg, cost.weights, best_hand_quality)

    counter = itertools.count()
    open_heap: List[Tuple[float, int, float, NodeKey]] = []
    g_scores: Dict[NodeKey, float] = {}
    nodes: Dict[NodeKey, SearchState] = {}
    came_from: Dict[NodeKey, Tuple[Optional[NodeKey], Optional[Move]]] = {}

    for ss, g in seeds:
        key: NodeKey = (ss.body.key(), ss.base, ss.mask)
        if key in g_scores and g_scores[key] <= g:
            continue
        g_scores[key] = g
        nodes[key] = ss
        came_from[key] = (None, None)
        f = g + h(ss, graph, floor, eff_heuristic_weight)
        heapq.heappush(open_heap, (f, next(counter), g, key))

    expansions = 0
    start_time = time.perf_counter()
    goal_key: Optional[NodeKey] = None

    while open_heap:
        if expansions >= max_expansions:
            break
        if time.perf_counter() - start_time > time_budget_s:
            break

        f, _, g, key = heapq.heappop(open_heap)
        if g > g_scores.get(key, float("inf")) + 1e-9:
            continue  # stale entry, a better path to this node was already found

        ss = nodes[key]
        if is_goal(
            graph,
            ss,
            matched_finish=matched_finish,
            require_foot_on=require_foot_on,
            require_coverage=require_coverage,
        ):
            goal_key = key
            break

        expansions += 1
        for nxt_ss, move, search_cost in _expand(
            ss, graph, cost, idx_of_hold_id, W, hand_hold_penalty=hand_hold_penalty, relaxed=relaxed
        ):
            nxt_key: NodeKey = (nxt_ss.body.key(), nxt_ss.base, nxt_ss.mask)
            nxt_g = g + search_cost
            if nxt_key in g_scores and g_scores[nxt_key] <= nxt_g + 1e-9:
                continue
            g_scores[nxt_key] = nxt_g
            nodes[nxt_key] = nxt_ss
            came_from[nxt_key] = (key, move)
            nxt_f = nxt_g + h(nxt_ss, graph, floor, eff_heuristic_weight)
            heapq.heappush(open_heap, (nxt_f, next(counter), nxt_g, nxt_key))

    if goal_key is None:
        return None

    # -- reconstruct path ---------------------------------------------------
    rev_keys = [goal_key]
    rev_moves: List[Move] = []
    cur = goal_key
    while came_from[cur][0] is not None:
        parent_key, move = came_from[cur]
        rev_moves.append(move)
        cur = parent_key
        rev_keys.append(cur)

    keys = list(reversed(rev_keys))
    moves = list(reversed(rev_moves))
    states = [nodes[k].body for k in keys]

    return states, moves


def solve(
    graph: ClimbGraph,
    cost: CostModel,
    *,
    heuristic_weight: float = 2.2,
    window: Optional[int] = None,
    max_expansions: int = 20_000,
    time_budget_s: float = 0.2,
    matched_finish: bool = True,
    require_foot_on: bool = True,
    n_foot_seeds: int = 5,
    hand_hold_penalty: Optional[Dict[int, float]] = None,
) -> Optional[Beta]:
    """Weighted A* search for a `Beta`. `None` if no goal is found within
    the bounds (exhausted or timed out) -- never raises, never returns a
    partial/non-goal beta.

    `hand_hold_penalty` (`{hold_id: extra_cost}`) is not part of the cost
    model -- it never touches `Move.cost`/`Beta.total_cost` -- it only
    biases *this search's* choices, for callers like `solve_k` that want a
    different hand sequence out of a re-solve, not a different opinion of
    what any given move actually costs.
    """
    result = _astar(
        graph,
        cost,
        heuristic_weight=heuristic_weight,
        window=window,
        max_expansions=max_expansions,
        time_budget_s=time_budget_s,
        matched_finish=matched_finish,
        require_foot_on=require_foot_on,
        n_foot_seeds=n_foot_seeds,
        hand_hold_penalty=hand_hold_penalty,
    )
    if result is None:
        return None
    states, moves = result
    return Beta(states=states, moves=moves, angle_deg=cost.angle_deg, profile=cost.profile)


def _solve_relaxed(
    graph: ClimbGraph,
    cost: CostModel,
    *,
    max_expansions: int,
    time_budget_s: float,
    matched_finish: bool,
    require_foot_on: bool,
    n_foot_seeds: int,
    hand_hold_penalty: Optional[Dict[int, float]] = None,
) -> Optional[Beta]:
    """Coverage-as-cost fallback for climbs the strict-order engine can't
    solve at all -- primarily the ~7% of real climbs (traverses,
    down-climbs) whose y-sorted hand order isn't how the climb is actually
    meant to be climbed, so no window size makes `solve`'s rolling
    `(base, mask)` legal-to-advance.

    Drops the progress coordinate entirely: `_astar(relaxed=True)` doesn't
    track or gate on coverage at all, so the goal is just "matched finish,
    foot on" and the state space shrinks to plain `BodyState` dedup -- a
    faster search than the strict-order one, not a slower one. Coverage
    isn't free, though: every hand-role hold the winning path never touched
    is charged `W_SKIP` and reported on the returned `Beta`
    (`relaxed`, `skipped_hold_ids`, `skip_penalty`) rather than silently
    dropped, so a caller can see exactly how much of the climb this
    fallback gave up on sequencing.
    """
    result = _astar(
        graph,
        cost,
        heuristic_weight=0.0,  # ignored when relaxed=True; passed for clarity
        window=None,
        max_expansions=max_expansions,
        time_budget_s=time_budget_s,
        matched_finish=matched_finish,
        require_foot_on=require_foot_on,
        n_foot_seeds=n_foot_seeds,
        hand_hold_penalty=hand_hold_penalty,
        relaxed=True,
    )
    if result is None:
        return None
    states, moves = result

    touched = {states[0].rh.hold_id, states[0].lh.hold_id}
    touched.update(m.to_hold for m in moves if m.limb in HANDS)
    touched.discard(None)
    all_hand_ids = {int(hid) for hid in graph.hold_id[graph.hand_ix]}
    skipped = tuple(sorted(all_hand_ids - touched))
    skip_penalty = cost.weights["W_SKIP"] * len(skipped)

    return Beta(
        states=states,
        moves=moves,
        angle_deg=cost.angle_deg,
        profile=cost.profile,
        relaxed=True,
        skipped_hold_ids=skipped,
        skip_penalty=skip_penalty,
    )


def solve_or_relax(
    graph: ClimbGraph,
    cost: CostModel,
    *,
    heuristic_weight: float = 2.2,
    window: Optional[int] = None,
    max_expansions: int = 20_000,
    time_budget_s: float = 0.2,
    matched_finish: bool = True,
    require_foot_on: bool = True,
    n_foot_seeds: int = 5,
    hand_hold_penalty: Optional[Dict[int, float]] = None,
    escalate: bool = True,
    escalated_heuristic_weight: float = 1.3,
    escalated_max_expansions: int = 150_000,
    escalated_time_budget_s: float = 2.0,
    relax: bool = True,
    relax_max_expansions: int = 60_000,
    relax_time_budget_s: float = 1.0,
) -> Optional[Beta]:
    """Three escalating tiers of `solve`, tried in order until one succeeds.

    This exists because of a gap measured on the real DB, not the one
    originally scoped: default-settings `solve()` only converges on
    ~21-33% of real climbs (see `test_core_search.py`'s module docstring),
    and the root cause is mostly the cost model's `tension`/`balance`/
    `foot_cut` terms routinely pushing real move costs 2-3x past what
    `heuristic_weight=2.2` compresses for within the default expansion/time
    budget -- not the y-gap/traverse case this fallback was originally
    scoped for. So there are two independent reasons `solve` can fail, and
    this tries a fix for each, cheapest first:

    1. **Fast** -- `solve` at the given (default: production/generation-loop)
       budget. Succeeds on the large majority of climbs; this is the path
       that has to stay fast for `core.generate`'s hot loop.
    2. **Escalated** -- the same strict-order engine, same seeds, same
       physics, just a far larger expansion/time budget and a heuristic
       weight closer to the proven-admissible floor. This is the fix for
       the search-budget majority: the climb *is* solvable by this engine,
       it just needs more room than the fast tier's budget gives it. Only
       run if the fast tier fails.
    3. **Relaxed** -- `_solve_relaxed`'s coverage-as-cost fallback, for the
       minority the strict-order engine cannot solve at *any* budget (real
       traverses/down-climbs). Only run if both prior tiers fail.

    Returns `None` only if all enabled tiers fail (or `escalate=relax=False`
    reduces this to plain `solve`).
    """
    beta = solve(
        graph,
        cost,
        heuristic_weight=heuristic_weight,
        window=window,
        max_expansions=max_expansions,
        time_budget_s=time_budget_s,
        matched_finish=matched_finish,
        require_foot_on=require_foot_on,
        n_foot_seeds=n_foot_seeds,
        hand_hold_penalty=hand_hold_penalty,
    )
    if beta is not None:
        return beta

    if escalate:
        beta = solve(
            graph,
            cost,
            heuristic_weight=escalated_heuristic_weight,
            window=window,
            max_expansions=escalated_max_expansions,
            time_budget_s=escalated_time_budget_s,
            matched_finish=matched_finish,
            require_foot_on=require_foot_on,
            n_foot_seeds=n_foot_seeds,
            hand_hold_penalty=hand_hold_penalty,
        )
        if beta is not None:
            return beta

    if relax:
        return _solve_relaxed(
            graph,
            cost,
            max_expansions=relax_max_expansions,
            time_budget_s=relax_time_budget_s,
            matched_finish=matched_finish,
            require_foot_on=require_foot_on,
            n_foot_seeds=n_foot_seeds,
            hand_hold_penalty=hand_hold_penalty,
        )

    return None


# ---------------------------------------------------------------------------
# Diverse betas (solve_k)
# ---------------------------------------------------------------------------

#: Below this Jaccard similarity on `hand_signature`, two betas are counted
#: as genuinely different climbing lines rather than the same hands with a
#: different foot smear.
DEFAULT_JACCARD_THRESHOLD = 0.7

#: Extra search-only cost added to a hand hold each time a `solve_k` attempt
#: uses it, so the next re-solve is nudged toward a different hand sequence
#: instead of reproducing the same one (see `solve_k`'s docstring for why
#: this replaces Yen's k-shortest-paths here).
PENALTY_STEP = 2.0


def hand_signature(beta: Beta) -> frozenset:
    """`{(limb, hold_id)}` for every hand move in `beta` -- an order-free
    fingerprint of *which hand went where*, used to tell two betas apart by
    their actual climbing line rather than by an incidental foot smear.
    """
    return frozenset((m.limb, m.to_hold) for m in beta.moves if m.limb in HANDS)


def _hand_jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def solve_k(
    graph: ClimbGraph,
    cost: CostModel,
    k: int = 5,
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    penalty_step: float = PENALTY_STEP,
    max_attempts: Optional[int] = None,
    **solve_or_relax_kwargs,
) -> List[Beta]:
    """Up to `k` diverse betas for the same hold set/angle, via penalized
    re-search.

    Not Yen's k-shortest-paths: on this graph, paths adjacent in cost
    overwhelmingly differ by one edge -- a foot placement -- so Yen
    reproduces the exact failure being fixed here, returning the same hand
    sequence `k` times with different smears (and costs O(k * L * A*), ~560
    ms at k=5, on top of that). Instead: solve, record the beta's
    `hand_signature`, add `penalty_step` to every hold that signature used,
    solve again. A candidate is only *kept* if it's dissimilar enough
    (`_hand_jaccard < jaccard_threshold`) from every beta already kept, but
    the penalty escalates on every attempt regardless -- so a near-duplicate
    still pushes the next attempt further away rather than being wasted.
    O(k * A*), ~40 ms at k=5 on a climb `solve` handles at its fast tier.

    Built on `solve_or_relax` (not bare `solve`) so a climb that only the
    escalated or relaxed tier can solve still benefits from diverse-beta
    generation instead of unconditionally returning `[]`; `**kwargs` pass
    straight through to it (`heuristic_weight`, budgets, etc.).
    """
    if k <= 0:
        return []
    max_attempts = max_attempts if max_attempts is not None else max(k * 4, 4)

    results: List[Beta] = []
    signatures: List[frozenset] = []
    penalty: Dict[int, float] = {}

    attempts = 0
    while len(results) < k and attempts < max_attempts:
        attempts += 1
        beta = solve_or_relax(graph, cost, hand_hold_penalty=dict(penalty), **solve_or_relax_kwargs)
        if beta is None:
            break  # exhausted at this penalty level -- more attempts won't help

        sig = hand_signature(beta)
        if all(_hand_jaccard(sig, kept) < jaccard_threshold for kept in signatures):
            results.append(beta)
            signatures.append(sig)

        for _, hold_id in sig:
            if hold_id is not None:
                penalty[hold_id] = penalty.get(hold_id, 0.0) + penalty_step

    return results
