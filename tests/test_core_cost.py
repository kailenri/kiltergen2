"""Tests for body state and the angle-conditioned cost model."""

import math
import random

import pytest

from config import MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING
from core import params
from core.cost import WEIGHT_NAME, CostModel, MoveContext, MoveScore, resolve_move_effects
from core.holds import HoldInfo
from core.physics import foot_cut_threshold, in_plane_pull_fraction
from core.state import (
    CUT,
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


def on(hold_id, x, y):
    return Contact(kind=ON_HOLD, hold_id=hold_id, xy=(x, y))


def base_state(angle_deg=40.0):
    """Hands overhead, feet planted, a normal mid-climb position."""
    return BodyState(
        rh=on(1, X_SPACING, 4.0 * X_SPACING),
        lh=on(2, -X_SPACING, 4.0 * X_SPACING),
        rf=on(3, X_SPACING, 0.0),
        lf=on(4, -X_SPACING, 0.0),
        angle_deg=angle_deg,
    )


#: Holds with no orientation data, so direction terms stay out of the way of
#: tests that are about something else.
BLANK_HOLDS = HoldInfo(data={})


# -- state ------------------------------------------------------------------


def test_state_is_hashable_and_usable_as_a_search_node():
    a, b = base_state(), base_state()
    assert a.key() == b.key()
    assert len({a.key(), b.key()}) == 1


def test_key_distinguishes_cut_from_planted_on_the_same_holds():
    """Same holds, different body situation -- must not collapse in a visited set."""
    planted = base_state()
    cut = planted.with_contact("RF", CUT_CONTACT)
    assert planted.key() != cut.key()


def test_flagging_foot_gives_stance_but_carries_no_load():
    state = base_state().with_contact(
        "LF", Contact(kind=FLAGGING, hold_id=None, xy=(-3.0 * X_SPACING, X_SPACING))
    )
    assert (-3.0 * X_SPACING, X_SPACING) in state.stance_points()
    assert (-3.0 * X_SPACING, X_SPACING) not in state.weighted_points()


def test_hip_falls_back_to_hanging_when_both_feet_are_cut():
    state = base_state().with_contact("RF", CUT_CONTACT).with_contact("LF", CUT_CONTACT)
    hip = state.hip()
    assert hip is not None
    # Hangs beneath the hands rather than returning None.
    assert hip[1] < 4.0 * X_SPACING


def test_with_contact_does_not_mutate_the_original():
    state = base_state()
    state.with_contact("RH", CUT_CONTACT)
    assert state.rh.kind == ON_HOLD


# -- the point of the whole exercise: angle changes the answer --------------


def test_same_move_costs_more_at_a_steeper_angle():
    """The one thing the legacy scorer structurally could not do."""
    target = (2.0 * X_SPACING, 6.0 * X_SPACING)
    costs = []
    for angle in (10.0, 30.0, 50.0, 65.0):
        model = CostModel(angle_deg=angle, holds=BLANK_HOLDS)
        total, _, _ = model.score_move(base_state(angle), "RH", target, 9)
        costs.append(total)
    assert costs == sorted(costs)
    assert costs[-1] > costs[0]


def test_tension_term_only_appears_once_the_board_is_steep():
    target = (2.0 * X_SPACING, 6.0 * X_SPACING)
    _, shallow, _ = CostModel(angle_deg=5.0, holds=BLANK_HOLDS).score_move(
        base_state(5.0), "RH", target, 9
    )
    _, steep, _ = CostModel(angle_deg=60.0, holds=BLANK_HOLDS).score_move(
        base_state(60.0), "RH", target, 9
    )
    assert "tension" not in shallow
    assert steep.get("tension", 0.0) > 0.0


def test_grip_cost_rises_with_angle():
    target = (2.0 * X_SPACING, 6.0 * X_SPACING)
    _, shallow, _ = CostModel(angle_deg=10.0, holds=BLANK_HOLDS).score_move(
        base_state(10.0), "RH", target, 9
    )
    _, steep, _ = CostModel(angle_deg=60.0, holds=BLANK_HOLDS).score_move(
        base_state(60.0), "RH", target, 9
    )
    assert steep["grip"] > shallow["grip"]


# -- move terms -------------------------------------------------------------


def test_short_move_is_free_on_the_reach_term():
    model = CostModel(angle_deg=30.0, holds=BLANK_HOLDS)
    near = (X_SPACING * 1.2, 4.2 * X_SPACING)
    _, breakdown, _ = model.score_move(base_state(30.0), "RH", near, 9)
    assert "reach" not in breakdown


def test_long_move_pays_a_reach_penalty():
    model = CostModel(angle_deg=30.0, holds=BLANK_HOLDS)
    far = (X_SPACING, 4.0 * X_SPACING + 0.95 * MAX_HAND_REACH)
    _, breakdown, _ = model.score_move(base_state(30.0), "RH", far, 9)
    assert breakdown.get("reach", 0.0) > 0.0


def test_big_reach_cuts_a_foot_sooner_on_a_steep_board():
    """Angle-derived cut threshold, not a fixed DYNAMIC_REACH_FACTOR."""
    # A reach sized to sit between the vertical and steep thresholds.
    reach = 0.72 * MAX_HAND_REACH
    target = (X_SPACING, 4.0 * X_SPACING + reach)

    _, shallow, shallow_fx = CostModel(angle_deg=0.0, holds=BLANK_HOLDS).score_move(
        base_state(0.0), "RH", target, 9
    )
    _, steep, steep_fx = CostModel(angle_deg=65.0, holds=BLANK_HOLDS).score_move(
        base_state(65.0), "RH", target, 9
    )
    assert "dynamic_cut" not in shallow_fx
    assert "foot_cut" in steep


def test_low_quality_hold_costs_more_grip_than_a_jug():
    target = (2.0 * X_SPACING, 6.0 * X_SPACING)
    jugs = HoldInfo(data={i: {"quality": 1.0, "confidence": 0.0} for i in range(1, 10)})
    crimps = HoldInfo(data={i: {"quality": 0.3, "confidence": 0.0} for i in range(1, 10)})
    _, good, _ = CostModel(angle_deg=40.0, holds=jugs).score_move(
        base_state(40.0), "RH", target, 9
    )
    _, bad, _ = CostModel(angle_deg=40.0, holds=crimps).score_move(
        base_state(40.0), "RH", target, 9
    )
    assert bad["grip"] > good["grip"]


def test_crossing_hands_is_penalised():
    model = CostModel(angle_deg=30.0, holds=BLANK_HOLDS)
    across = (-3.0 * X_SPACING, 5.0 * X_SPACING)  # RH landing left of LH
    _, breakdown, effects = model.score_move(base_state(30.0), "RH", across, 9)
    assert "cross" in breakdown
    assert "cross" in effects


def test_hold_direction_term_fades_out_on_steep_ground():
    """Edge orientation matters on a slab and barely does on a roof."""
    # An edge whose usable direction is square to the pull is penalty-free,
    # so use one rotated to be awkward.
    holds = HoldInfo(data={9: {"quality": 0.7, "direction_deg": 90.0, "confidence": 1.0}})
    target = (X_SPACING, 6.0 * X_SPACING)

    _, slab, _ = CostModel(angle_deg=5.0, holds=holds).score_move(
        base_state(5.0), "RH", target, 9
    )
    _, roof, _ = CostModel(angle_deg=85.0, holds=holds).score_move(
        base_state(85.0), "RH", target, 9
    )
    assert slab.get("hold_direction", 0.0) > roof.get("hold_direction", 0.0)


def test_unmapped_hold_skips_the_direction_term_entirely():
    """No orientation data means no term, not an assumed value."""
    model = CostModel(angle_deg=30.0, holds=HoldInfo(data={}))
    _, breakdown, _ = model.score_move(
        base_state(30.0), "RH", (X_SPACING, 6.0 * X_SPACING), 999
    )
    assert "hold_direction" not in breakdown


def test_breakdown_sums_to_total():
    model = CostModel(angle_deg=45.0, holds=BLANK_HOLDS)
    total, breakdown, _ = model.score_move(
        base_state(45.0), "RH", (2.0 * X_SPACING, 6.0 * X_SPACING), 9
    )
    assert total == pytest.approx(sum(breakdown.values()))


# -- profiles ---------------------------------------------------------------


def test_beginner_and_expert_profiles_price_a_big_move_differently():
    """Same graph, different weights -- that is the whole mechanism."""
    target = (X_SPACING, 4.0 * X_SPACING + 0.95 * MAX_HAND_REACH)
    state = base_state(40.0)
    beginner, _, _ = CostModel(40.0, profile="beginner", holds=BLANK_HOLDS).score_move(
        state, "RH", target, 9
    )
    expert, _, _ = CostModel(40.0, profile="expert", holds=BLANK_HOLDS).score_move(
        state, "RH", target, 9
    )
    assert beginner > expert


def test_unknown_profile_falls_back_to_neutral_weights():
    a = CostModel(40.0, profile="nonsense", holds=BLANK_HOLDS)
    b = CostModel(40.0, profile="intermediate", holds=BLANK_HOLDS)
    assert a.weights == b.weights


# -- effects ----------------------------------------------------------------


def test_dynamic_cut_removes_the_opposite_foot():
    state = base_state(60.0)
    nxt = resolve_move_effects(state, "RH", (0.0, 0.0), ("dynamic_cut",))
    assert nxt.lf.kind == CUT
    assert nxt.rf.kind == ON_HOLD


def test_no_effects_leaves_the_state_alone():
    state = base_state()
    assert resolve_move_effects(state, "RH", (0.0, 0.0), ()) is state


# -- beta projection --------------------------------------------------------


def test_limb_tracks_partition_the_move_list():
    moves = [
        Move(limb="RH", from_hold=1, to_hold=5, cost=1.0),
        Move(limb="LF", from_hold=4, to_hold=6, cost=0.5),
        Move(limb="RH", from_hold=5, to_hold=7, cost=2.0),
    ]
    beta = Beta(states=[base_state()] * 3, moves=moves, angle_deg=40.0)
    tracks = beta.limb_tracks()
    assert [i for i, _ in tracks["RH"]] == [0, 2]
    assert [i for i, _ in tracks["LF"]] == [1]
    assert tracks["LH"] == []
    total = sum(len(t) for t in tracks.values())
    assert total == len(moves)


def test_crux_is_the_most_expensive_move():
    moves = [
        Move(limb="RH", from_hold=1, to_hold=5, cost=1.0),
        Move(limb="LH", from_hold=2, to_hold=6, cost=3.5),
        Move(limb="RF", from_hold=3, to_hold=7, cost=0.2),
    ]
    beta = Beta(states=[base_state()] * 3, moves=moves, angle_deg=40.0)
    idx, move = beta.crux()
    assert idx == 1
    assert move.limb == "LH"


def test_move_reports_its_dominant_reason():
    move = Move(
        limb="RH",
        from_hold=1,
        to_hold=5,
        cost=3.0,
        breakdown={"grip": 0.5, "tension": 2.0, "reach": 0.5},
    )
    assert move.dominant_reason() == "tension"


def test_move_without_a_breakdown_has_no_reason():
    assert Move(limb="RH", from_hold=1, to_hold=5).dominant_reason() is None


# -- prepare() / score_prepared() refactor -----------------------------------
#
# `score_move` is now a thin wrapper around `prepare()` (the target-independent
# statics solve) followed by `score_prepared()` (the per-target geometry). The
# tests below pin down that split: an independent reimplementation of the
# pre-refactor formulas for parity, the weights-dot-features invariant the
# calibration step will rely on, the new flat `move` term, and that a single
# `MoveContext` is safe to reuse across many candidate targets.


def _legacy_reach_cost(model, state, limb, target_xy):
    """Mirrors the pre-refactor `CostModel._reach_cost` body exactly."""
    current = state.contact(limb)
    max_reach = MAX_HAND_REACH if limb in HANDS else MAX_FOOT_REACH
    if current.xy is None:
        return (0.0, 0.0)
    dist = math.hypot(target_xy[0] - current.xy[0], target_xy[1] - current.xy[1])
    fraction = dist / max_reach if max_reach > 0 else 0.0
    if fraction <= params.REACH_FREE_FRACTION:
        return (0.0, fraction)
    over = (fraction - params.REACH_FREE_FRACTION) / (1.0 - params.REACH_FREE_FRACTION)
    return (model.weights["W_REACH"] * over * over, fraction)


def _legacy_hold_direction_cost(model, landed, limb, target_xy, target_hold_id):
    """Mirrors the pre-refactor `CostModel._hold_direction_cost` body exactly.

    Uses `landed.hip()` (the limb already at its target), not the
    transitional (limb-removed) hip the refactor now uses. The two agree
    whenever at least one foot is providing stance -- which every state the
    random search below generates guarantees -- so this stays a faithful
    parity check without re-triggering the documented target-independent-hip
    approximation.
    """
    hip = landed.hip()
    if hip is None:
        return 0.0
    shoulder = (hip[0], hip[1] + 2.0 * X_SPACING)
    pull_u = target_xy[0] - shoulder[0]
    pull_v = target_xy[1] - shoulder[1]
    alignment = model.holds.pull_alignment(target_hold_id, pull_u, pull_v)
    if alignment is None:
        return 0.0
    plane_share = in_plane_pull_fraction(model.angle_deg)
    return model.weights["W_HOLD_DIR"] * (1.0 - alignment) * plane_share


def _legacy_cross_cost(model, landed, limb, target_xy):
    """Mirrors the pre-refactor `CostModel._cross_cost` body exactly."""
    if limb in HANDS:
        other = "LH" if limb == "RH" else "RH"
    else:
        other = "LF" if limb == "RF" else "RF"
    other_xy = landed.contact(other).xy
    if other_xy is None:
        return 0.0
    pad = 0.5 * X_SPACING
    crossed = (
        target_xy[0] < other_xy[0] - pad
        if limb in ("RH", "RF")
        else target_xy[0] > other_xy[0] + pad
    )
    if not crossed:
        return 0.0
    loads = landed.loads()
    relief = loads.foot_engagement if loads is not None else 0.0
    return model.weights["W_CROSS"] * (1.0 - 0.5 * relief)


def legacy_score_move(model, state, limb, target_xy, target_hold_id):
    """Independent reimplementation of the pre-refactor `score_move`.

    Written straight from the formulas in the pre-refactor `core/cost.py`
    (not by calling any of the refactored code), plus the new flat `move`
    term this same change is adding -- so a match against the refactored
    `score_move` is a genuine before/after parity check, not a tautology.
    """
    transitional = state.with_contact(limb, CUT_CONTACT)
    landed = state.with_contact(
        limb, Contact(kind=ON_HOLD, hold_id=target_hold_id, xy=target_xy)
    )
    breakdown = {}
    effects = []

    loads = transitional.loads(foot_quality=model._foot_quality(transitional))
    if loads is not None:
        grip_quality = model._hand_quality(transitional)
        n_hands = max(1, len(transitional.weighted_points(HANDS)))
        per_hand = loads.hand_load / n_hands
        breakdown["grip"] = model.weights["W_GRIP"] * per_hand / max(0.15, grip_quality)
        if loads.tension_demand > 0.0:
            breakdown["tension"] = model.weights["W_TENSION"] * loads.tension_demand
        if not loads.balanced:
            breakdown["balance"] = model.weights["W_BALANCE"]
        if not loads.feet_secure and transitional.weighted_points(FEET):
            breakdown["foot_cut"] = model.weights["W_FOOT_CUT"]
            effects.append("feet_blow_off")

    reach_cost, reach_fraction = _legacy_reach_cost(model, state, limb, target_xy)
    if reach_cost > 0.0:
        breakdown["reach"] = reach_cost

    if limb in HANDS and reach_fraction > foot_cut_threshold(model.angle_deg):
        if "foot_cut" not in breakdown:
            breakdown["foot_cut"] = model.weights["W_FOOT_CUT"]
            effects.append("dynamic_cut")

    if limb in HANDS:
        dir_cost = _legacy_hold_direction_cost(model, landed, limb, target_xy, target_hold_id)
        if dir_cost > 0.0:
            breakdown["hold_direction"] = dir_cost

    cross_cost = _legacy_cross_cost(model, landed, limb, target_xy)
    if cross_cost > 0.0:
        breakdown["cross"] = cross_cost
        effects.append("cross")

    breakdown["move"] = model.weights["W_MOVE"]

    return (sum(breakdown.values()), breakdown, tuple(effects))


def _random_contact(rng, hold_ids):
    kind = rng.choice([ON_HOLD, FLAGGING, CUT])
    if kind == CUT:
        return CUT_CONTACT
    xy = (rng.uniform(-4.0, 4.0) * X_SPACING, rng.uniform(-1.0, 8.0) * X_SPACING)
    if kind == ON_HOLD:
        return Contact(kind=ON_HOLD, hold_id=rng.choice(hold_ids), xy=xy)
    return Contact(kind=FLAGGING, hold_id=None, xy=xy)


def _random_state(rng, hold_ids):
    """A random, physically-arbitrary state.

    At least one foot always keeps stance (ON_HOLD or FLAGGING). Hip is
    derived purely from feet whenever any foot has stance, so this sidesteps
    the one documented case where the refactor's target-independent hip
    (from the transitional state) and the pre-refactor per-target hip (from
    the landed state) can disagree -- both feet cut *and* a hand move -- and
    keeps this a fair apples-to-apples parity check.
    """
    angle = rng.choice([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0])
    rh = _random_contact(rng, hold_ids)
    lh = _random_contact(rng, hold_ids)
    rf = _random_contact(rng, hold_ids)
    lf_kind_forced_on = rf.kind == CUT
    if lf_kind_forced_on:
        xy = (rng.uniform(-4.0, 4.0) * X_SPACING, rng.uniform(-1.0, 8.0) * X_SPACING)
        lf = rng.choice(
            [
                Contact(kind=ON_HOLD, hold_id=rng.choice(hold_ids), xy=xy),
                Contact(kind=FLAGGING, hold_id=None, xy=xy),
            ]
        )
    else:
        lf = _random_contact(rng, hold_ids)
    return BodyState(rh=rh, lh=lh, rf=rf, lf=lf, angle_deg=angle)


def _random_holds(rng):
    data = {}
    for hid in range(1, 30):
        if rng.random() < 0.3:
            continue  # leave some holds unmapped, exercising the fallback
        data[hid] = {
            "quality": rng.uniform(0.2, 1.0),
            "direction_deg": rng.uniform(0.0, 180.0),
            "confidence": rng.uniform(0.0, 1.0),
        }
    return HoldInfo(data=data)


def test_prepare_and_score_prepared_matches_legacy_formulas_over_random_moves():
    """Before/after parity: refactored `score_move` vs. an independent
    reimplementation of the pre-refactor formulas, over ~500 random cases
    spanning angle, limb, contact kind (including FLAGGING and CUT), and
    hold id/quality.
    """
    rng = random.Random(20260806)
    hold_ids = list(range(1, 30)) + [None]
    n_checked = 0
    for _ in range(500):
        holds = _random_holds(rng)
        state = _random_state(rng, hold_ids)
        model = CostModel(angle_deg=state.angle_deg, holds=holds)
        limb = rng.choice(LIMBS)
        target_xy = (rng.uniform(-4.0, 4.0) * X_SPACING, rng.uniform(-1.0, 8.0) * X_SPACING)
        target_hold_id = rng.choice(hold_ids)

        expected_total, expected_breakdown, expected_effects = legacy_score_move(
            model, state, limb, target_xy, target_hold_id
        )
        got_total, got_breakdown, got_effects = model.score_move(
            state, limb, target_xy, target_hold_id
        )

        assert got_total == pytest.approx(expected_total, abs=1e-9)
        assert got_breakdown.keys() == expected_breakdown.keys()
        for key in expected_breakdown:
            assert got_breakdown[key] == pytest.approx(expected_breakdown[key], abs=1e-9)
        assert set(got_effects) == set(expected_effects)
        n_checked += 1

    assert n_checked == 500


def test_score_move_is_prepare_plus_score_prepared():
    """`score_move` really is the two-step split, not a parallel code path."""
    model = CostModel(angle_deg=45.0, holds=BLANK_HOLDS)
    state = base_state(45.0)
    target = (2.0 * X_SPACING, 6.0 * X_SPACING)

    ctx = model.prepare(state, "RH")
    assert isinstance(ctx, MoveContext)
    score = model.score_prepared(ctx, target, 9)
    assert isinstance(score, MoveScore)

    total, breakdown, effects = model.score_move(state, "RH", target, 9)
    assert total == score.total
    assert breakdown == score.breakdown
    assert effects == score.effects


# -- weights (dot) features invariant ----------------------------------------


def _assert_weights_dot_features(model, ctx, score):
    """`breakdown[k] == weights[WEIGHT_NAME[k]] * features[k]` for every key,
    and `total == sum(...)` -- the invariant the calibration step depends on.
    """
    assert score.breakdown.keys() == score.features.keys()
    for key, feature in score.features.items():
        weight_name = WEIGHT_NAME[key]
        expected = model.weights[weight_name] * feature
        assert score.breakdown[key] == pytest.approx(expected, abs=1e-9), key
    recomposed = sum(
        model.weights[WEIGHT_NAME[k]] * v for k, v in score.features.items()
    )
    assert score.total == pytest.approx(recomposed, abs=1e-9)
    assert score.total == pytest.approx(sum(score.breakdown.values()), abs=1e-9)


def test_total_equals_weights_dot_features_across_angles_and_limbs():
    """The linear-in-weights invariant the future calibration step relies on,
    swept across angles, hand and foot moves, and scenarios that trigger
    foot_cut, cross, hold_direction, and balance.
    """
    oriented_holds = HoldInfo(
        data={9: {"quality": 0.7, "direction_deg": 90.0, "confidence": 1.0}}
    )
    triggered_keys = set()

    for angle in (0.0, 20.0, 40.0, 60.0, 85.0):
        model = CostModel(angle_deg=angle, holds=oriented_holds)
        state = base_state(angle)

        cases = [
            ("RH", (X_SPACING, 6.0 * X_SPACING), 9),  # hold_direction candidate
            ("LF", (1.5 * X_SPACING, 1.0 * X_SPACING), 10),  # ordinary foot move
            ("RH", (-3.0 * X_SPACING, 5.0 * X_SPACING), 9),  # crossing hand move
            (  # big reach -> reach + possible dynamic foot_cut
                "RH",
                (X_SPACING, 4.0 * X_SPACING + 0.95 * MAX_HAND_REACH),
                9,
            ),
        ]
        for limb, target_xy, target_hold_id in cases:
            ctx = model.prepare(state, limb)
            score = model.score_prepared(ctx, target_xy, target_hold_id)
            _assert_weights_dot_features(model, ctx, score)
            triggered_keys.update(score.features.keys())

    # An imbalanced pose, constructed directly (see test_core_physics.py's
    # balance tests for the same shape of scenario).
    imbalanced = BodyState(
        rh=on(1, 8.0 * X_SPACING, 1.0 * X_SPACING),
        lh=CUT_CONTACT,
        rf=on(3, -0.2 * X_SPACING, 0.0),
        lf=on(4, 0.2 * X_SPACING, 0.0),
        angle_deg=30.0,
    )
    model = CostModel(angle_deg=30.0, holds=BLANK_HOLDS)
    ctx = model.prepare(imbalanced, "LH")
    score = model.score_prepared(ctx, (2.0 * X_SPACING, 3.0 * X_SPACING), 5)
    _assert_weights_dot_features(model, ctx, score)
    triggered_keys.update(score.features.keys())

    # A steep, statically-insecure pose that isn't relieved by feet.
    steep = base_state(75.0)
    model = CostModel(angle_deg=75.0, holds=BLANK_HOLDS)
    ctx = model.prepare(steep, "RH")
    score = model.score_prepared(ctx, (2.0 * X_SPACING, 6.0 * X_SPACING), 9)
    _assert_weights_dot_features(model, ctx, score)
    triggered_keys.update(score.features.keys())

    assert {
        "grip",
        "move",
        "reach",
        "hold_direction",
        "cross",
        "balance",
    } <= triggered_keys


# -- the flat W_MOVE term -----------------------------------------------------


def test_move_term_is_always_present_and_never_scales():
    """`move` is a flat charge -- exactly `W_MOVE`, feature 1.0, on both a
    cheap short move and an expensive dynamic one."""
    model = CostModel(angle_deg=60.0, holds=BLANK_HOLDS)
    state = base_state(60.0)

    cheap_ctx = model.prepare(state, "RH")
    cheap = model.score_prepared(cheap_ctx, (X_SPACING * 1.1, 4.1 * X_SPACING), 9)

    expensive_ctx = model.prepare(state, "RH")
    expensive = model.score_prepared(
        expensive_ctx, (-3.0 * X_SPACING, 4.0 * X_SPACING + 0.98 * MAX_HAND_REACH), 9
    )

    assert cheap.breakdown["move"] == model.weights["W_MOVE"]
    assert expensive.breakdown["move"] == model.weights["W_MOVE"]
    assert cheap.features["move"] == 1.0
    assert expensive.features["move"] == 1.0
    # The two moves cost very differently overall, but not because of `move`.
    assert cheap.total != pytest.approx(expensive.total)
    assert model.weights["W_MOVE"] == pytest.approx(params.W_MOVE)


# -- MoveContext is reusable across targets ----------------------------------


def test_prepare_is_reusable_across_multiple_targets_without_mutation():
    model = CostModel(angle_deg=45.0, holds=BLANK_HOLDS)
    state = base_state(45.0)
    ctx = model.prepare(state, "RH")

    base_breakdown_before = dict(ctx.base_breakdown)
    base_features_before = dict(ctx.base_features)

    near = (X_SPACING * 1.2, 4.2 * X_SPACING)
    far = (X_SPACING, 4.0 * X_SPACING + 0.95 * MAX_HAND_REACH)

    near_score = model.score_prepared(ctx, near, 9)
    # ctx unchanged after the first call.
    assert ctx.base_breakdown == base_breakdown_before
    assert ctx.base_features == base_features_before

    far_score = model.score_prepared(ctx, far, 9)
    # ctx still unchanged after a second call with a different target.
    assert ctx.base_breakdown == base_breakdown_before
    assert ctx.base_features == base_features_before

    # Results are independent and match what a direct score_move gives.
    assert "reach" not in near_score.breakdown
    assert far_score.breakdown.get("reach", 0.0) > 0.0
    assert near_score.total != pytest.approx(far_score.total)

    expected_near_total, expected_near_breakdown, _ = model.score_move(state, "RH", near, 9)
    expected_far_total, expected_far_breakdown, _ = model.score_move(state, "RH", far, 9)
    assert near_score.total == pytest.approx(expected_near_total)
    assert near_score.breakdown == expected_near_breakdown
    assert far_score.total == pytest.approx(expected_far_total)
    assert far_score.breakdown == expected_far_breakdown

    # A third call with yet another target still reuses the same ctx cleanly.
    third = (2.0 * X_SPACING, 6.0 * X_SPACING)
    third_score = model.score_prepared(ctx, third, 9)
    assert third_score.total > 0.0
    assert ctx.base_breakdown == base_breakdown_before
