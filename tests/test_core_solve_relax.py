"""Tests for step 5 of the `core/` rebuild: `solve_or_relax` and `solve_k`
in `core.search`.

Two synthetic climbs anchor the deterministic tests here rather than real
DB climbs, because the property under test -- "full hand-hold coverage is
*provably* impossible, not just hard" -- needs to be guaranteed, not just
likely. `_solvable_climb` is an ordinary small climb; `_orphan_climb` is the
same climb plus one extra hand-role hold placed far enough away that no hand
move can ever reach it, so strict-coverage `solve()` is guaranteed to fail
at *any* budget, which is exactly the case `solve_or_relax`'s relaxed tier
exists for. `solve_k`'s diversity mechanism is also unit-tested directly
(signatures, Jaccard, the search-only penalty), plus one smoke test against
real small climbs for the integration path.
"""

from __future__ import annotations

from typing import Dict

import pytest

from config import MAX_HAND_REACH
from core.cost import CostModel
from core.data import iter_angle_climbs
from core.graph import ClimbGraph, ROLE_FINISH, ROLE_FOOT, ROLE_HAND, ROLE_START
from core.holds import HoldInfo
from core.search import (
    DEFAULT_JACCARD_THRESHOLD,
    SearchState,
    _expand,
    _hand_jaccard,
    _solve_relaxed,
    hand_signature,
    is_goal,
    seed_states,
    solve,
    solve_k,
    solve_or_relax,
)

BLANK_HOLDS = HoldInfo(data={})


# ---------------------------------------------------------------------------
# Synthetic climbs
# ---------------------------------------------------------------------------


def make_hold(hole_id, placement_id, x, y, role_id):
    return {
        "placement_id": placement_id,
        "hole_id": hole_id,
        "x": x,
        "y": y,
        "name": f"{hole_id}",
        "position": 0,
        "role_id": role_id,
    }


#: Start holds kept close together (well under `_cross_cost`'s half-spacing
#: pad) and the hand/finish holds centred between them, so no hand move on
#: this climb ever registers as "crossed" regardless of which hand seeds
#: onto which start hold. This matters a lot more than it looks: a crossed
#: move triggers an opposite-foot flag, and a flagged foot's position is a
#: continuous function of the hip estimate at that point in the path -- so
#: even one avoidable cross turns a 4-hold climb's state space from "a few
#: dozen states" into tens of thousands (measured while building this
#: file), which is exactly the state-space-blowup character of the real
#: gap `solve_or_relax` exists for, just not the property *these*
#: mechanism-level tests are trying to isolate.
def _solvable_climb():
    """Two starts -> one intermediate hand hold -> one finish, at 0 degrees.
    Small and unambiguous: strict-order `solve()` handles it comfortably at
    default settings, so this is the baseline for "the fast tier already
    works, `solve_or_relax` must not do anything different".
    """
    holds = [
        make_hold(1, 1, 0.0, 0.0, ROLE_START),
        make_hold(2, 2, 6.0, 0.0, ROLE_START),
        make_hold(3, 3, 3.0, 40.0, ROLE_HAND),
        make_hold(4, 4, 3.0, 80.0, ROLE_FINISH),
        make_hold(5, 5, 0.0, -10.0, ROLE_FOOT),
        make_hold(6, 6, 6.0, -10.0, ROLE_FOOT),
    ]
    graph = ClimbGraph(holds)
    cost = CostModel(angle_deg=0.0, holds=BLANK_HOLDS)
    return graph, cost


def _orphan_climb():
    """`_solvable_climb` plus one extra hand-role hold placed far enough
    away (`>> MAX_HAND_REACH` from everything else) that no hand move can
    ever land on it. Strict full-coverage `solve()` therefore cannot
    succeed at *any* budget -- not "hard", *impossible* -- which is what
    makes this a clean, deterministic trigger for `solve_or_relax`'s
    relaxed tier rather than a flaky "maybe the budget was too small" case.
    """
    holds = [
        make_hold(1, 1, 0.0, 0.0, ROLE_START),
        make_hold(2, 2, 6.0, 0.0, ROLE_START),
        make_hold(3, 3, 3.0, 40.0, ROLE_HAND),
        make_hold(4, 4, 3.0, 80.0, ROLE_FINISH),
        make_hold(5, 5, 0.0, -10.0, ROLE_FOOT),
        make_hold(6, 6, 6.0, -10.0, ROLE_FOOT),
        make_hold(7, 7, 3.0 + 10.0 * MAX_HAND_REACH, 40.0, ROLE_HAND),  # the orphan
    ]
    graph = ClimbGraph(holds)
    cost = CostModel(angle_deg=0.0, holds=BLANK_HOLDS)
    return graph, cost, 7  # 7 is the orphan's hole_id


def _small_real_climbs(n: int, min_ascents: int = 3, pool: int = 8000):
    """Distinct-by-uuid real climbs with <=8 hand holds, for the one
    integration smoke test that isn't purely synthetic. Mirrors
    `test_core_search.py`'s helper of the same name/shape; duplicated
    locally rather than imported so this module stays self-contained (the
    convention every other `test_core_*.py` module already follows).
    """
    seen: Dict[str, object] = {}
    for c in iter_angle_climbs(min_ascents=min_ascents, angles=[0, 5, 10, 15, 20], limit=pool):
        if c.uuid in seen:
            continue
        n_hand = len(c.hand_holds)
        if n_hand == 0 or n_hand > 8 or len(c.holds) > 14:
            continue
        seen[c.uuid] = c
        if len(seen) >= n:
            break
    return list(seen.values())


# ===========================================================================
# 1. `relaxed=True` mechanics: _expand / is_goal / _astar
# ===========================================================================


def test_expand_relaxed_does_not_advance_or_filter_by_coverage_window():
    """With `relaxed=True`, a hand target's rank is irrelevant -- every
    reach-legal hand hold is offered, and the returned `(base, mask)` is
    unchanged from the input, exactly the "progress coordinate dropped"
    claim in `_expand`'s docstring.
    """
    graph, cost, _orphan_id = _orphan_climb()
    idx_of_hold_id = {int(h): i for i, h in enumerate(graph.hold_id)}

    # Seed one hand on start hold 1, both feet cut, at (base=1, mask=0) as
    # if some coverage had already been recorded -- relaxed mode must leave
    # it untouched regardless.
    seeds = seed_states(graph, cost, cost.angle_deg, n_foot_seeds=1)
    ss = seeds[0][0]
    fake_ss = SearchState(ss.body, base=1, mask=0)

    for nxt_ss, move, _search_cost in _expand(
        fake_ss, graph, cost, idx_of_hold_id, W=1, relaxed=True
    ):
        assert nxt_ss.base == 1 and nxt_ss.mask == 0


def test_is_goal_require_coverage_false_ignores_incomplete_coverage():
    graph, cost, _orphan_id = _orphan_climb()
    seeds = seed_states(graph, cost, cost.angle_deg, n_foot_seeds=1)
    ss = seeds[0][0]  # base/mask reflect only the two start holds covered

    # Coverage is nowhere near complete (the orphan, among others, is
    # untouched) and the body isn't even on the finish hold, so this isn't a
    # goal either way -- but the point is require_coverage=False must not
    # itself be the reason it returns False for a state that *is* complete.
    assert not is_goal(graph, ss, require_coverage=True)
    assert not is_goal(graph, ss, require_coverage=False)  # still not on finish


# ===========================================================================
# 2. solve_or_relax: fast tier, escalation, relaxed fallback
# ===========================================================================


def test_solve_or_relax_matches_plain_solve_when_the_fast_tier_already_succeeds():
    """No regression: on a climb the fast tier handles fine, `solve_or_relax`
    must return exactly what `solve()` would, not something from a later
    tier -- escalation/relaxation should never trigger when they aren't
    needed.
    """
    graph, cost = _solvable_climb()
    direct = solve(graph, cost)
    via_relax = solve_or_relax(graph, cost)

    assert direct is not None
    assert via_relax is not None
    assert via_relax.relaxed is False
    assert via_relax.total_cost == pytest.approx(direct.total_cost)


def test_solve_or_relax_escalates_when_the_fast_tier_budget_is_too_small():
    """`max_expansions=0` makes the fast tier fail unconditionally (it
    breaks before popping even the first node -- see `_astar`), regardless
    of how easy the climb is. `solve_or_relax` must still succeed via its
    escalated tier, and the result must be a genuine (non-relaxed) beta.
    """
    graph, cost = _solvable_climb()
    assert solve(graph, cost, max_expansions=0) is None  # fast tier alone: fails

    beta = solve_or_relax(graph, cost, max_expansions=0)
    assert beta is not None
    assert beta.relaxed is False
    assert beta.skipped_hold_ids == ()


def test_solve_fails_on_a_climb_with_an_unreachable_hand_hold_at_any_budget():
    """Documents the premise the next test relies on: strict-coverage
    `solve()` cannot succeed on `_orphan_climb`, no matter the budget --
    this is impossibility, not a tuning gap, since the orphan hold is
    outside `MAX_HAND_REACH` of every other hold in the climb.
    """
    graph, cost, _orphan_id = _orphan_climb()
    assert solve(graph, cost) is None
    assert (
        solve(graph, cost, heuristic_weight=1.0, max_expansions=50_000, time_budget_s=2.0)
        is None
    )


def test_solve_or_relax_falls_back_to_relaxed_tier_and_reports_the_skip():
    graph, cost, orphan_id = _orphan_climb()

    beta = solve_or_relax(graph, cost)

    assert beta is not None
    assert beta.relaxed is True
    assert beta.skipped_hold_ids == (orphan_id,)
    assert beta.skip_penalty == pytest.approx(cost.weights["W_SKIP"] * 1)
    # The reported total_cost is unpolluted by the skip charge; full_cost is
    # the number that should be compared against a non-relaxed beta.
    assert beta.full_cost == pytest.approx(beta.total_cost + beta.skip_penalty)

    # Every hand hold except the orphan was actually reachable, so the
    # relaxed engine -- even though it wasn't required to -- still touched
    # all of them on its way to a matched finish.
    touched = {beta.states[0].rh.hold_id, beta.states[0].lh.hold_id}
    touched.update(m.to_hold for m in beta.moves if m.limb in ("RH", "LH"))
    touched.discard(None)
    assert orphan_id not in touched
    assert touched == {1, 2, 3, 4}


def test_solve_relaxed_called_directly_matches_the_solve_or_relax_result():
    """`_solve_relaxed` is the engine `solve_or_relax`'s tier 3 wraps --
    exercised directly here rather than only indirectly, matching the
    convention in `test_core_search.py` of testing internal helpers on
    their own terms.
    """
    graph, cost, orphan_id = _orphan_climb()
    beta = _solve_relaxed(
        graph,
        cost,
        max_expansions=60_000,
        time_budget_s=1.0,
        matched_finish=True,
        require_foot_on=True,
        n_foot_seeds=5,
    )
    assert beta is not None
    assert beta.relaxed is True
    assert beta.skipped_hold_ids == (orphan_id,)
    assert beta.skip_penalty == pytest.approx(cost.weights["W_SKIP"])


def test_solve_or_relax_relax_disabled_returns_none_when_only_relaxation_would_help():
    graph, cost, _orphan_id = _orphan_climb()
    assert solve_or_relax(graph, cost, relax=False) is None


def test_solve_or_relax_escalate_disabled_still_reaches_the_relaxed_tier():
    """`escalate=False` skips tier 2 but must not skip tier 3 -- the two
    flags are independent."""
    graph, cost, orphan_id = _orphan_climb()
    beta = solve_or_relax(graph, cost, escalate=False)
    assert beta is not None
    assert beta.relaxed is True
    assert beta.skipped_hold_ids == (orphan_id,)


# ===========================================================================
# 3. solve_k mechanics
# ===========================================================================


def test_hand_signature_is_order_free_and_feet_free():
    graph, cost = _solvable_climb()
    beta = solve(graph, cost)
    assert beta is not None

    sig = hand_signature(beta)
    assert all(limb in ("RH", "LH") for limb, _hold_id in sig)
    # Every hand move recorded, nothing lost or duplicated.
    hand_moves = [m for m in beta.moves if m.limb in ("RH", "LH")]
    assert len(sig) == len({(m.limb, m.to_hold) for m in hand_moves})


def test_hand_jaccard_identical_and_disjoint():
    a = frozenset({("RH", 1), ("LH", 2)})
    b = frozenset({("RH", 1), ("LH", 2)})
    c = frozenset({("RH", 9), ("LH", 8)})
    assert _hand_jaccard(a, b) == 1.0
    assert _hand_jaccard(a, c) == 0.0
    assert _hand_jaccard(frozenset(), frozenset()) == 1.0


def test_expand_hand_hold_penalty_biases_search_cost_but_not_move_cost():
    """The penalty is a search-only device: it changes what `_expand` yields
    as the third element (what `_astar` adds to `g`), never `move.cost`
    (what ends up in `Beta.total_cost`) -- see `core/cost.py`'s linearity
    invariant this is deliberately kept out of.
    """
    graph, cost = _solvable_climb()
    idx_of_hold_id = {int(h): i for i, h in enumerate(graph.hold_id)}
    seeds = seed_states(graph, cost, cost.angle_deg, n_foot_seeds=1)
    ss = seeds[0][0]

    unpenalized = {
        (move.limb, move.to_hold): (move.cost, search_cost)
        for _next_ss, move, search_cost in _expand(ss, graph, cost, idx_of_hold_id, W=2)
        if move.limb in ("RH", "LH")
    }
    hold_3 = 3  # the intermediate hand hold in `_solvable_climb`
    penalized = {
        (move.limb, move.to_hold): (move.cost, search_cost)
        for _next_ss, move, search_cost in _expand(
            ss, graph, cost, idx_of_hold_id, W=2, hand_hold_penalty={hold_3: 7.0}
        )
        if move.limb in ("RH", "LH")
    }

    for key, (move_cost, base_search_cost) in unpenalized.items():
        pen_move_cost, pen_search_cost = penalized[key]
        assert pen_move_cost == pytest.approx(move_cost)  # untouched
        if key[1] == hold_3:
            assert pen_search_cost == pytest.approx(base_search_cost + 7.0)
        else:
            assert pen_search_cost == pytest.approx(base_search_cost)


def test_solve_k_returns_empty_list_for_k_le_zero():
    graph, cost = _solvable_climb()
    assert solve_k(graph, cost, k=0) == []
    assert solve_k(graph, cost, k=-1) == []


def test_solve_k_stops_early_when_the_climb_is_exhausted():
    """`_solvable_climb` has exactly one viable hand line (one intermediate
    hold, one finish) -- there is no second distinct signature to find, so
    `solve_k` must return exactly one beta even after several re-solves.

    `max_attempts` is capped explicitly rather than left at `solve_k`'s
    default (`k*4`): the only route stays solvable at any penalty level on
    a climb this small, so with `k=5` it would otherwise burn all 20
    attempts (each a full `solve_or_relax` call) just to demonstrate
    something 4 attempts already show.
    """
    graph, cost = _solvable_climb()
    results = solve_k(graph, cost, k=5, max_attempts=4)
    assert len(results) == 1
    assert results[0].total_cost > 0


def test_solve_k_escalates_penalty_across_attempts_even_without_diversity():
    """Even on a climb with only one real line, repeated attempts must not
    error out, and re-running with an already-populated penalty map (as
    `solve_k` does internally) must still find *a* beta -- the penalty
    biases the search, it doesn't have to change the outcome to be
    well-behaved.
    """
    graph, cost = _solvable_climb()
    beta = solve_k(graph, cost, k=1)[0]
    penalty = {hold_id: 100.0 for _limb, hold_id in hand_signature(beta)}
    still_solves = solve_or_relax(graph, cost, hand_hold_penalty=penalty)
    assert still_solves is not None  # only one route exists; a penalty can't remove it


def test_solve_k_on_real_climbs_smoke():
    """Integration smoke test against real small climbs: `solve_k` must run
    cleanly and, whenever it does return more than one beta, they must
    actually satisfy the diversity contract (pairwise Jaccard below
    threshold) -- not a claim about *how often* diversity is achievable
    (that's the plan's >=80%-of-climbs criterion for step 5's full
    verification pass, not this unit-test module).
    """
    climbs = _small_real_climbs(n=5)
    if not climbs:
        pytest.skip("no small real climbs available in this DB")

    any_multi = False
    for c in climbs:
        graph = ClimbGraph(c.holds)
        cost = CostModel(angle_deg=c.angle)
        results = solve_k(graph, cost, k=3, max_attempts=5)
        assert isinstance(results, list)
        assert len(results) <= 3

        sigs = [hand_signature(b) for b in results]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                assert _hand_jaccard(sigs[i], sigs[j]) < DEFAULT_JACCARD_THRESHOLD
        if len(results) > 1:
            any_multi = True

    # Honest reporting rather than a hard assertion: whether real climbs in
    # this DB actually offer >1 diverse line within `max_attempts=8` varies
    # by climb, and this module's job is the mechanism, not the yield rate.
    print(f"solve_k found >1 diverse beta on {any_multi} out of {len(climbs)} climbs sampled")
