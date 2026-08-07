"""Tests for the angle-independent hold-set precompute.

`ClimbGraph` is built once per hold set and reused across every board angle,
profile, and beta -- so what matters here is that its reach/distance
computation is *exactly* right (checked against an independent brute-force
reference), that its indexing is deterministic (a later A* search keys its
state space on positions within `hand_order`), and that it is fast enough to
build for every proposal a hold-set optimizer will throw at it.
"""

import math
import time

import numpy as np
import pytest

from config import MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING
from core.data import iter_angle_climbs
from core.graph import ClimbGraph, ROLE_FINISH, ROLE_FOOT, ROLE_HAND, ROLE_START


# -- shared fixtures ----------------------------------------------------------


def _distinct_climbs(n=20, min_ascents=5, pool=2000):
    """Pull `n` distinct hold sets (by uuid) from real climb data."""
    seen = {}
    for c in iter_angle_climbs(min_ascents=min_ascents, limit=pool):
        if c.uuid not in seen:
            seen[c.uuid] = c
        if len(seen) >= n:
            break
    return list(seen.values())


def brute_force_reach(holds, max_reach):
    """Loop-based reference reach matrix -- what the vectorized one is checked against."""
    n = len(holds)
    out = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = math.hypot(holds[i]["x"] - holds[j]["x"], holds[i]["y"] - holds[j]["y"])
            out[i][j] = d <= max_reach
    return out


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


# -- 1. reach matrices match brute-force -------------------------------------


def test_reach_matrices_match_brute_force_on_real_climbs():
    """Vectorized dist/reach must agree exactly with a loop-based reference."""
    climbs = _distinct_climbs(20)
    assert len(climbs) == 20
    for climb in climbs:
        graph = ClimbGraph(climb.holds)
        # brute force must be computed over the same canonical order as the
        # graph, since graph.hand_reach etc. are indexed that way.
        ordered_holds = list(graph.holds)
        expected_hand = brute_force_reach(ordered_holds, MAX_HAND_REACH)
        expected_foot = brute_force_reach(ordered_holds, MAX_FOOT_REACH)
        n = len(ordered_holds)
        for i in range(n):
            for j in range(n):
                assert graph.hand_reach[i, j] == expected_hand[i][j]
                assert graph.foot_reach[i, j] == expected_foot[i][j]


# -- 2. hand_order ordering ----------------------------------------------------


def test_hand_order_respects_y_then_x_then_hold_id():
    """A synthetic set with an intentional y-tie exercises the full tiebreak chain."""
    holds = [
        make_hold(hole_id=30, placement_id=300, x=5, y=10, role_id=ROLE_HAND),
        make_hold(hole_id=10, placement_id=100, x=1, y=10, role_id=ROLE_HAND),
        make_hold(hole_id=20, placement_id=200, x=1, y=10, role_id=ROLE_HAND),  # y,x tie with hole_id=10
        make_hold(hole_id=5, placement_id=50, x=0, y=0, role_id=ROLE_START),
        make_hold(hole_id=40, placement_id=400, x=2, y=20, role_id=ROLE_FINISH),
    ]
    graph = ClimbGraph(holds)
    ordered_hold_ids = [int(graph.hold_id[i]) for i in graph.hand_order]
    # y=0 (hole 5) first, then y=10 x=1 tie broken by hole_id (10 before 20),
    # then y=10 x=5 (hole 30), then y=20 (hole 40).
    assert ordered_hold_ids == [5, 10, 20, 30, 40]


def test_hand_order_ties_broken_by_x_before_hold_id():
    holds = [
        make_hold(hole_id=99, placement_id=1, x=0, y=0, role_id=ROLE_HAND),
        make_hold(hole_id=1, placement_id=2, x=5, y=0, role_id=ROLE_HAND),
    ]
    graph = ClimbGraph(holds)
    ordered_hold_ids = [int(graph.hold_id[i]) for i in graph.hand_order]
    # Same y; the lower x (0) must come first even though its hold_id (99)
    # is larger -- x outranks hold_id in the tiebreak order.
    assert ordered_hold_ids == [99, 1]


# -- 3. rank is the inverse of hand_order -------------------------------------


def test_rank_is_correct_inverse_of_hand_order():
    climbs = _distinct_climbs(5)
    for climb in climbs:
        graph = ClimbGraph(climb.holds)
        for pos, hold_idx in enumerate(graph.hand_order):
            hold_id = int(graph.hold_id[hold_idx])
            assert graph.rank[hold_id] == pos
        assert len(graph.rank) == len(graph.hand_order)


# -- 4. connectivity_ok --------------------------------------------------------


def test_connectivity_ok_false_when_a_hand_hold_is_stranded():
    """A hand hold placed far above everything below it can't be reached."""
    holds = [
        make_hold(hole_id=1, placement_id=1, x=0, y=0, role_id=ROLE_START),
        make_hold(hole_id=2, placement_id=2, x=1, y=1 * X_SPACING, role_id=ROLE_HAND),
        make_hold(hole_id=3, placement_id=3, x=0, y=2 * X_SPACING, role_id=ROLE_HAND),
        # Miles above everything else -- far beyond MAX_HAND_REACH from any
        # other hand hold.
        make_hold(
            hole_id=4,
            placement_id=4,
            x=0,
            y=2 * X_SPACING + 20 * MAX_HAND_REACH,
            role_id=ROLE_FINISH,
        ),
    ]
    graph = ClimbGraph(holds)
    assert graph.connectivity_ok() is False


def test_connectivity_ok_true_on_a_smooth_real_climb():
    """A small, smoothly-progressing real hold set should pass the check."""
    climbs = _distinct_climbs(30)
    # Pick the climb with the smallest hand-hold count and the smallest
    # max y-gap -- an easy, smoothly-spaced climb the beam search would
    # obviously find a way through.
    best = None
    best_key = None
    for climb in climbs:
        graph = ClimbGraph(climb.holds)
        if graph.hand_order.size < 3:
            continue
        key = (graph.max_y_gap(), graph.hand_order.size)
        if best_key is None or key < best_key:
            best_key = key
            best = graph
    assert best is not None
    assert best.connectivity_ok() is True


def test_connectivity_ok_trivially_true_with_one_or_no_hand_holds():
    holds = [make_hold(hole_id=1, placement_id=1, x=0, y=0, role_id=ROLE_START)]
    graph = ClimbGraph(holds)
    assert graph.connectivity_ok() is True

    graph_empty_hands = ClimbGraph(
        [make_hold(hole_id=1, placement_id=1, x=0, y=0, role_id=ROLE_FOOT)]
    )
    assert graph_empty_hands.connectivity_ok() is True


# -- 5. hand_targets / foot_targets ---------------------------------------------


def test_targets_never_include_self_and_match_reach_rows():
    climbs = _distinct_climbs(10)
    for climb in climbs:
        graph = ClimbGraph(climb.holds)
        n = len(graph.holds)
        for i in range(n):
            assert i not in graph.hand_targets[i]
            assert i not in graph.foot_targets[i]

            expected_hand = set(np.nonzero(graph.hand_reach[i])[0]) & set(graph.hand_ix.tolist())
            expected_hand.discard(i)
            assert set(graph.hand_targets[i].tolist()) == expected_hand

            expected_foot = set(np.nonzero(graph.foot_reach[i])[0]) & set(graph.foot_ix.tolist())
            expected_foot.discard(i)
            assert set(graph.foot_targets[i].tolist()) == expected_foot

            # sorted ascending
            assert list(graph.hand_targets[i]) == sorted(graph.hand_targets[i].tolist())
            assert list(graph.foot_targets[i]) == sorted(graph.foot_targets[i].tolist())


# -- 6. degenerate input --------------------------------------------------------


def test_start_and_finish_only_climb_does_not_crash():
    holds = [
        make_hold(hole_id=1, placement_id=1, x=0, y=0, role_id=ROLE_START),
        make_hold(hole_id=2, placement_id=2, x=1, y=5 * X_SPACING, role_id=ROLE_FINISH),
    ]
    graph = ClimbGraph(holds)
    n = len(holds)
    assert graph.dist.shape == (n, n)
    assert graph.hand_reach.shape == (n, n)
    assert graph.foot_reach.shape == (n, n)
    assert graph.hand_order.size == 2
    assert len(graph.hand_targets) == n
    assert len(graph.foot_targets) == n
    assert len(graph.band) == 2
    assert isinstance(graph.connectivity_ok(), bool)
    assert graph.max_y_gap() == pytest.approx(5 * X_SPACING)


def test_feet_use_any_hold_default_includes_hand_holds_as_foot_targets():
    """Kilter convention (confirmed via legacy is_valid_foot_transition, which
    applies no role filter beyond the reach matrix): feet may use any placed
    hold, not just role_id == 15.
    """
    holds = [
        make_hold(hole_id=1, placement_id=1, x=0, y=0, role_id=ROLE_START),
        make_hold(hole_id=2, placement_id=2, x=1, y=1, role_id=ROLE_HAND),
        make_hold(hole_id=3, placement_id=3, x=0, y=2, role_id=ROLE_FOOT),
    ]
    graph = ClimbGraph(holds, feet_use_any_hold=True)
    assert set(graph.foot_ix.tolist()) == {0, 1, 2}

    graph_strict = ClimbGraph(holds, feet_use_any_hold=False)
    assert set(graph_strict.foot_ix.tolist()) == {2}


# -- 7. performance -------------------------------------------------------------


def test_construction_is_fast_for_fifty_real_climbs():
    """Sanity check, not a strict benchmark: building 50 graphs should be near-instant."""
    climbs = _distinct_climbs(50)
    assert len(climbs) == 50
    start = time.perf_counter()
    for climb in climbs:
        ClimbGraph(climb.holds)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
