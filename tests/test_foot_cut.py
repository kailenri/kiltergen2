"""Tests for dynamic-move foot-cut + replant rule."""
import json

from config import (
    DYNAMIC_REACH_FACTOR,
    FOOT_CUT_ENABLED,
    MAX_HAND_REACH,
    X_SPACING,
)
from kinematics import (
    choose_cut_foot,
    is_dynamic_hand_move,
)
from sequence_generator import ClimbSequenceGenerator


# ---------------------------------------------------------------------------
# Pure unit tests of kinematics
# ---------------------------------------------------------------------------

def test_dynamic_long_reach_triggers():
    """A reach beyond DYNAMIC_REACH_FACTOR * MAX_HAND_REACH is dynamic."""
    prev = (0.0, 0.0)
    far = (DYNAMIC_REACH_FACTOR * MAX_HAND_REACH + 5.0, 0.0)
    feet = [(-X_SPACING, -50.0), (X_SPACING, -50.0)]
    assert is_dynamic_hand_move(prev, far, feet, 'RH', other_hand_xy=(-10.0, 0.0))


def test_short_reach_not_dynamic():
    """A short, balanced reach is not dynamic."""
    prev = (0.0, 0.0)
    near = (X_SPACING, 5.0)
    feet = [(-X_SPACING, -50.0), (X_SPACING, -50.0)]
    assert not is_dynamic_hand_move(prev, near, feet, 'RH', other_hand_xy=(-10.0, 0.0))


def test_choose_cut_foot_opposite_side():
    """Opposite-side foot is preferred (RH dyno -> LF cuts)."""
    feet = {'RF': (10.0, 0.0), 'LF': (-10.0, 0.0)}
    assert choose_cut_foot('RH', feet) == 'LF'
    assert choose_cut_foot('LH', feet) == 'RF'


def test_choose_cut_foot_fallback_when_opposite_missing():
    """If opposite-side foot is already off, fall back to the other foot."""
    feet = {'RF': (10.0, 0.0), 'LF': None}
    assert choose_cut_foot('RH', feet) == 'RF'


def test_choose_cut_foot_none_when_no_feet():
    feet = {'RF': None, 'LF': None}
    assert choose_cut_foot('RH', feet) is None


# ---------------------------------------------------------------------------
# Beam-state integration tests using the real generator
# ---------------------------------------------------------------------------

def _load_sample_generator():
    with open('sample_climbs.json') as f:
        data = json.load(f)
    holds = data['results'][0]['best_sequence']['holds']
    return ClimbSequenceGenerator(holds), holds


def _make_state(gen, hand_ids, foot_ids, cut_feet=frozenset(), sequence=None):
    """Build a synthetic beam state at the same shape _expand_states expects."""
    limbs = {
        'RH': gen.hold_dict[hand_ids['RH']] if hand_ids.get('RH') is not None else None,
        'LH': gen.hold_dict[hand_ids['LH']] if hand_ids.get('LH') is not None else None,
        'RF': gen.hold_dict[foot_ids['RF']] if foot_ids.get('RF') is not None else None,
        'LF': gen.hold_dict[foot_ids['LF']] if foot_ids.get('LF') is not None else None,
    }
    return {
        'sequence': list(sequence or []),
        'limbs': limbs,
        'cut_feet': frozenset(cut_feet),
        'score': 0.0,
    }


def test_replant_gate_blocks_hand_expansions():
    """When cut_feet is non-empty, _expand_states emits no hand moves."""
    if not FOOT_CUT_ENABLED:
        return  # rule disabled — nothing to test
    gen, holds = _load_sample_generator()
    starts = [h['hole_id'] for h in holds if h.get('role_id') == 12]
    feet_holds = [h['hole_id'] for h in holds if h.get('role_id') == 15] or [holds[0]['hole_id']]
    if len(starts) < 2 or not feet_holds:
        return

    state = _make_state(
        gen,
        hand_ids={'RH': starts[0], 'LH': starts[1]},
        foot_ids={'RF': feet_holds[0], 'LF': None},
        cut_feet={'LF'},
        sequence=[
            {'limb': 'RH', 'hold': starts[0]},
            {'limb': 'LH', 'hold': starts[1]},
        ],
    )

    expanded = gen._expand_states([state], beam_width=4)
    # Every expansion must move the cut foot (LF), never a hand or the other foot.
    assert expanded, "expected at least one replant expansion"
    for s in expanded:
        last = s['sequence'][-1]
        assert last['limb'] == 'LF', (
            f"replant gate violated: emitted {last['limb']} while LF was cut"
        )


def test_replant_clears_cut_state():
    """After a replant expansion the cut foot is cleared from cut_feet."""
    if not FOOT_CUT_ENABLED:
        return
    gen, holds = _load_sample_generator()
    starts = [h['hole_id'] for h in holds if h.get('role_id') == 12]
    feet_holds = [h['hole_id'] for h in holds if h.get('role_id') == 15] or [holds[0]['hole_id']]
    if len(starts) < 2 or not feet_holds:
        return

    state = _make_state(
        gen,
        hand_ids={'RH': starts[0], 'LH': starts[1]},
        foot_ids={'RF': feet_holds[0], 'LF': None},
        cut_feet={'LF'},
        sequence=[
            {'limb': 'RH', 'hold': starts[0]},
            {'limb': 'LH', 'hold': starts[1]},
        ],
    )

    expanded = gen._expand_states([state], beam_width=4)
    for s in expanded:
        assert 'LF' not in s['cut_feet']


def test_evaluate_sequence_reports_recovery_metric():
    """evaluate_sequence emits foot_tension_recovery in details."""
    gen, holds = _load_sample_generator()
    starts = [h['hole_id'] for h in holds if h.get('role_id') == 12]
    finishes = [h['hole_id'] for h in holds if h.get('role_id') == 14]
    if len(starts) < 2 or not finishes:
        return
    seq = [
        {'limb': 'RH', 'hold': starts[0]},
        {'limb': 'LH', 'hold': starts[1]},
        {'limb': 'RH', 'hold': finishes[0]},
    ]
    out = gen.evaluate_sequence(seq)
    assert 'foot_tension_recovery' in out['details']
