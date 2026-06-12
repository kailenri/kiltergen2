"""Unit tests for phantom flag state (Phase E)."""

import math

import pytest

from kinematics import (
    choose_flag_foot,
    compute_flag_position,
    is_crossing_hand_move,
)
from sequence_generator import ClimbSequenceGenerator
from viz import classify_dynamic_moves, compute_bodies_along_sequence


# -- kinematics helpers -----------------------------------------------------

def test_is_crossing_hand_move_rh_crosses_left():
    # RH at x=8 crossing left past LH at x=24 (gap > pad).
    assert is_crossing_hand_move((8, 80), (24, 80), 'RH') is True


def test_is_crossing_hand_move_lh_crosses_right():
    # LH at x=40 crossing right past RH at x=24.
    assert is_crossing_hand_move((40, 80), (24, 80), 'LH') is True


def test_is_crossing_hand_move_no_cross_normal_reach():
    # RH reaching right while LH stays left — not a cross.
    assert is_crossing_hand_move((80, 80), (24, 80), 'RH') is False


def test_is_crossing_hand_move_no_other_hand():
    # Cannot cross what isn't there.
    assert is_crossing_hand_move((8, 80), None, 'RH') is False


def test_choose_flag_foot_picks_opposite():
    feet = {'RF': (40, 50), 'LF': (24, 50)}
    assert choose_flag_foot('RH', feet) == 'LF'
    assert choose_flag_foot('LH', feet) == 'RF'


def test_choose_flag_foot_returns_none_when_target_unplanted():
    feet = {'RF': (40, 50), 'LF': None}
    # RH wants LF to flag, but LF isn't planted.
    assert choose_flag_foot('RH', feet) is None


def test_compute_flag_position_opposite_of_hand():
    hip = (30, 60)
    hand = (60, 80)  # up and right
    flag_xy = compute_flag_position(hip, hand, 'LF')
    # Flag should be on the opposite side of the hand relative to hip.
    assert flag_xy[0] < hip[0]
    assert flag_xy[1] < hip[1]


# -- evaluate_sequence integration -----------------------------------------

def _synthetic_climb_for_cross():
    """Tight holds so a cross reach stays within MAX_HAND_REACH."""
    return [
        {'hole_id': 1, 'x': 40, 'y': 80, 'role_id': 12, 'position': 1},   # start RH
        {'hole_id': 2, 'x': 24, 'y': 80, 'role_id': 12, 'position': 2},   # start LH
        {'hole_id': 5, 'x': 32, 'y': 100, 'role_id': 14, 'position': 5}, # finish
        {'hole_id': 8, 'x': 8,  'y': 80, 'role_id': 13, 'position': 8},  # cross target
        {'hole_id': 10, 'x': 28, 'y': 50, 'role_id': 15, 'position': 10},  # LF
        {'hole_id': 11, 'x': 44, 'y': 50, 'role_id': 15, 'position': 11},  # RF
    ]


def test_evaluate_fires_flag_quality_on_cross_with_planted_feet():
    gen = ClimbSequenceGenerator(_synthetic_climb_for_cross())
    seq = [
        {'limb': 'RH', 'hold': 1},   # RH=(40,80)
        {'limb': 'LH', 'hold': 2},   # LH=(24,80)
        {'limb': 'RF', 'hold': 11},
        {'limb': 'LF', 'hold': 10},
        {'limb': 'RH', 'hold': 8},   # RH→(8,80): cross past LH
    ]
    ev = gen.evaluate_sequence(seq)
    assert ev['details']['flag_quality'] > 0


def test_evaluate_no_flag_when_no_cross():
    gen = ClimbSequenceGenerator(_synthetic_climb_for_cross())
    seq = [
        {'limb': 'RH', 'hold': 1},
        {'limb': 'LH', 'hold': 2},
        {'limb': 'RF', 'hold': 11},
        {'limb': 'LF', 'hold': 10},
        # Just continue with no cross.
    ]
    ev = gen.evaluate_sequence(seq)
    assert ev['details']['flag_quality'] == pytest.approx(0.0)


# -- viz / body integration ------------------------------------------------

def test_classify_dynamic_moves_emits_flag_limb_field():
    holds = _synthetic_climb_for_cross()
    seq = [
        {'limb': 'RH', 'hold': 1},
        {'limb': 'LH', 'hold': 2},
        {'limb': 'RF', 'hold': 11},
        {'limb': 'LF', 'hold': 10},
        {'limb': 'RH', 'hold': 8},
    ]
    tags = classify_dynamic_moves(holds, seq)
    assert all('flag_limb' in t and 'is_unflag' in t for t in tags)
    # The crossing move (index 4) should have flag_limb='LF'.
    assert tags[4]['flag_limb'] == 'LF'


def test_compute_bodies_renders_flag_on_cross():
    holds = _synthetic_climb_for_cross()
    seq = [
        {'limb': 'RH', 'hold': 1},
        {'limb': 'LH', 'hold': 2},
        {'limb': 'RF', 'hold': 11},
        {'limb': 'LF', 'hold': 10},
        {'limb': 'RH', 'hold': 8},
    ]
    bodies = compute_bodies_along_sequence(holds, seq)
    assert len(bodies) == 5
    final = bodies[-1]
    # After the cross, LF should be in a flag (lf_flag set, lf None).
    assert final.lf is None
    assert final.lf_flag is not None
