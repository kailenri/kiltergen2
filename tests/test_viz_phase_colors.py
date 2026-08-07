"""Unit tests for classify_dynamic_moves phase field and _PHASE_COLORS presence.

Verifies:
  - Every tag dict has a 'phase' key with one of the four valid values.
  - A synthetic dynamic move gets phase='dynamic'.
  - A synthetic replant move gets phase='recovery'.
  - A static hand move gets phase='static'.
  - _PHASE_COLORS has entries for all four phases.
"""
import pytest


VALID_PHASES = {'static', 'committing', 'dynamic', 'recovery'}


def _make_hold(hole_id, x, y, role_id):
    return {'hole_id': hole_id, 'x': float(x), 'y': float(y), 'role_id': role_id}


def _make_move(limb, hold_id, ttype=None):
    m = {'limb': limb, 'hold': hold_id}
    if ttype:
        m['transition_type'] = ttype
    return m


def test_phase_field_present_on_every_tag():
    from viz import classify_dynamic_moves

    holds = [
        _make_hold(1,  0.0, 0.0, 12),  # start
        _make_hold(2, 18.0, 0.0, 12),  # start
        _make_hold(3,  0.0, 19.0, 15), # foot RF
        _make_hold(4, 18.0, 19.0, 15), # foot LF
        _make_hold(5,  9.0, 38.0, 13), # hand
        _make_hold(6,  9.0, 57.0, 14), # finish
    ]
    sequence = [
        _make_move('RH', 1, 'start_hand'),
        _make_move('LH', 2, 'start_hand'),
        _make_move('RF', 3, 'start_foot'),
        _make_move('LF', 4, 'start_foot'),
        _make_move('RH', 5, 'hand_move'),
        _make_move('RH', 6, 'finish'),
    ]
    tags = classify_dynamic_moves(holds, sequence)
    assert len(tags) == len(sequence)
    for i, tag in enumerate(tags):
        assert 'phase' in tag, f"Move {i} missing 'phase' field"
        assert tag['phase'] in VALID_PHASES, (
            f"Move {i} has invalid phase '{tag['phase']}'"
        )


def test_dynamic_move_tagged_as_dynamic():
    """A move flagged as is_dynamic must get phase='dynamic'."""
    from viz import classify_dynamic_moves

    # Use transition_type override path since replicating kinematics in a
    # unit test is expensive.
    holds = [
        _make_hold(1,  0.0, 0.0, 12),
        _make_hold(2, 18.0, 0.0, 12),
        _make_hold(3,  0.0, 19.0, 15),
        _make_hold(4, 18.0, 19.0, 15),
        _make_hold(5, 36.0, 60.0, 13),
    ]
    sequence = [
        _make_move('RH', 1, 'start_hand'),
        _make_move('LH', 2, 'start_hand'),
        _make_move('RF', 3, 'start_foot'),
        _make_move('LF', 4, 'start_foot'),
        _make_move('RH', 5, 'dyno'),  # dyno -> dynamic
    ]
    tags = classify_dynamic_moves(holds, sequence)
    assert tags[-1]['phase'] == 'dynamic', (
        f"Expected phase='dynamic' for dyno move, got '{tags[-1]['phase']}'"
    )


def test_replant_tagged_as_recovery():
    from viz import classify_dynamic_moves

    holds = [
        _make_hold(1,  0.0, 0.0, 12),
        _make_hold(2, 18.0, 0.0, 12),
        _make_hold(3,  0.0, 19.0, 15),
        _make_hold(4, 18.0, 19.0, 15),
    ]
    sequence = [
        _make_move('RH', 1, 'start_hand'),
        _make_move('LH', 2, 'start_hand'),
        _make_move('RF', 3, 'start_foot'),
        _make_move('LF', 4, 'replant'),  # replant -> recovery
    ]
    tags = classify_dynamic_moves(holds, sequence)
    assert tags[-1]['phase'] == 'recovery', (
        f"Expected phase='recovery' for replant, got '{tags[-1]['phase']}'"
    )


def test_static_hand_move_tagged_as_static():
    from viz import classify_dynamic_moves

    holds = [
        _make_hold(1,  0.0, 0.0, 12),
        _make_hold(2, 18.0, 0.0, 12),
        _make_hold(3,  0.0, 19.0, 15),
        _make_hold(4, 18.0, 19.0, 15),
        _make_hold(5,  9.0, 38.0, 13),
    ]
    sequence = [
        _make_move('RH', 1, 'start_hand'),
        _make_move('LH', 2, 'start_hand'),
        _make_move('RF', 3, 'start_foot'),
        _make_move('LF', 4, 'start_foot'),
        _make_move('RH', 5, 'hand_move'),  # routine hand move -> static
    ]
    tags = classify_dynamic_moves(holds, sequence)
    assert tags[-1]['phase'] == 'static', (
        f"Expected phase='static' for routine hand move, got '{tags[-1]['phase']}'"
    )


def test_phase_colors_defined_for_all_phases():
    from viz import _PHASE_COLORS
    for phase in VALID_PHASES:
        assert phase in _PHASE_COLORS, f"_PHASE_COLORS missing entry for '{phase}'"
        color = _PHASE_COLORS[phase]
        assert color.startswith('#') and len(color) == 7, (
            f"_PHASE_COLORS['{phase}'] is not a valid hex color: {color!r}"
        )


def test_back_step_tagged_as_committing():
    from viz import classify_dynamic_moves

    holds = [
        _make_hold(1,  0.0, 0.0, 12),
        _make_hold(2, 18.0, 0.0, 12),
        _make_hold(3,  0.0, 19.0, 15),
        _make_hold(4, 18.0, 19.0, 15),
        _make_hold(5,  0.0, 38.0, 15),  # foot target
    ]
    sequence = [
        _make_move('RH', 1, 'start_hand'),
        _make_move('LH', 2, 'start_hand'),
        _make_move('RF', 3, 'start_foot'),
        _make_move('LF', 4, 'start_foot'),
        _make_move('RF', 5, 'back_step'),  # back_step -> committing
    ]
    tags = classify_dynamic_moves(holds, sequence)
    assert tags[-1]['phase'] == 'committing', (
        f"Expected phase='committing' for back_step, got '{tags[-1]['phase']}'"
    )
