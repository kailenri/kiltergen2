"""Unit tests for CV-derived hold quality + alignment scoring (Phase D)."""

import math

import pytest

from sequence_generator import (
    ClimbSequenceGenerator,
    get_hold_direction,
    get_hold_quality,
    load_hold_orientations,
)


def _synthetic_climb():
    """Tight synthetic board for deterministic evaluation."""
    return [
        {'hole_id': 1, 'x': 40, 'y': 80, 'role_id': 12, 'position': 1},   # start RH
        {'hole_id': 2, 'x': 24, 'y': 80, 'role_id': 12, 'position': 2},   # start LH
        {'hole_id': 3, 'x': 16, 'y': 80, 'role_id': 13, 'position': 3},
        {'hole_id': 4, 'x': 48, 'y': 80, 'role_id': 13, 'position': 4},
        {'hole_id': 5, 'x': 32, 'y': 100, 'role_id': 14, 'position': 5}, # finish
        {'hole_id': 10, 'x': 28, 'y': 50, 'role_id': 15, 'position': 10}, # LF
        {'hole_id': 11, 'x': 44, 'y': 50, 'role_id': 15, 'position': 11}, # RF
    ]


def test_quality_fallback_neutral_for_unknown_hold():
    # Hole id that won't exist in the sidecar.
    assert get_hold_quality(99999999) == pytest.approx(0.7)


def test_quality_in_valid_range_for_known_holds():
    oris = load_hold_orientations()
    if not oris:
        pytest.skip("hold_orientations.json missing — run scripts/build_hold_orientations.py")
    # Spot-check first 20 known holds.
    for hid, entry in list(oris.items())[:20]:
        q = get_hold_quality(hid)
        assert 0.5 <= q <= 1.0, f"quality {q} for hold {hid} out of [0.5, 1.0]"


def test_direction_returns_none_below_confidence():
    # Unknown holds always return None.
    assert get_hold_direction(99999999) is None


def test_direction_returned_in_valid_degree_range():
    oris = load_hold_orientations()
    if not oris:
        pytest.skip("hold_orientations.json missing")
    found = False
    for hid in oris.keys():
        deg = get_hold_direction(hid)
        if deg is not None:
            assert -360.0 < deg < 360.0
            found = True
            break
    assert found, "no holds passed the confidence threshold"


def test_evaluate_emits_hold_quality_and_alignment_keys():
    gen = ClimbSequenceGenerator(_synthetic_climb())
    seq = [
        {'limb': 'RH', 'hold': 1},
        {'limb': 'LH', 'hold': 2},
        {'limb': 'RF', 'hold': 11},
        {'limb': 'LF', 'hold': 10},
        {'limb': 'RH', 'hold': 4},
    ]
    ev = gen.evaluate_sequence(seq)
    assert 'hold_quality' in ev['details']
    assert 'hold_alignment' in ev['details']
    # quality should be > 0 since start/finish role weights kick in.
    assert ev['details']['hold_quality'] > 0


def test_alignment_zero_when_no_feet_planted():
    """Without planted feet there's no shoulder estimate, alignment skipped."""
    gen = ClimbSequenceGenerator(_synthetic_climb())
    # Only hand moves — no feet.
    seq = [
        {'limb': 'RH', 'hold': 1},
        {'limb': 'LH', 'hold': 2},
        {'limb': 'RH', 'hold': 4},
        {'limb': 'LH', 'hold': 3},
    ]
    ev = gen.evaluate_sequence(seq)
    assert ev['details']['hold_alignment'] == pytest.approx(0.0)
