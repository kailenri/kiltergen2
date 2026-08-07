"""Regression: ensure the trained v8 LSTM emits foot moves at inference.

Pre-Phase-A the entire training data was effectively hand-only
(<0.05% foot tokens) and the generator never produced feet at inference.
This guard fails loudly if a future change regresses that property.

Skips silently when the v8 model or vocab is not yet built so the suite
remains green in fresh clones.
"""

import os
import random

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'subset_v8_model.pth')
VOCAB_PATH = os.path.join(REPO_ROOT, 'data', 'vocab_subset_5k_v8.json')
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'training_sequences_subset_5k_v8.json')


def _missing_artifacts():
    return [p for p in (MODEL_PATH, VOCAB_PATH, DATA_PATH) if not os.path.exists(p)]


@pytest.fixture(scope='module')
def v8_generator():
    missing = _missing_artifacts()
    if missing:
        pytest.skip(f"v8 artifacts not built: {missing}")
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    from lstm import ClimbGenerator
    gen = ClimbGenerator(DATA_PATH, vocab_path=VOCAB_PATH)
    assert gen.load_model(MODEL_PATH), "v8 model failed to load"
    return gen


def test_v8_emits_foot_in_aggregate(v8_generator):
    """Across a small batch of generations, at least one sequence must contain
    a foot move. Loose threshold to absorb stochastic sampling."""
    rng = random.Random(0)
    cids = sorted(v8_generator.climb_data.keys())
    sample = rng.sample(cids, k=min(6, len(cids)))

    total_seqs = 0
    seqs_with_feet = 0
    for cid in sample:
        try:
            outs = v8_generator.generate(climb_id=cid, num_sequences=1,
                                         temperature=0.8, max_length=30)
        except Exception:
            continue
        if not outs:
            continue
        seq = outs[0].get('sequence') or []
        if not seq:
            continue
        total_seqs += 1
        if any(m.get('limb') in ('RF', 'LF') for m in seq):
            seqs_with_feet += 1

    assert total_seqs > 0, "v8 produced zero sequences across sample"
    assert seqs_with_feet >= 1, (
        f"v8 regression: 0/{total_seqs} generated sequences contained any foot move; "
        f"Phase-A beam fix may have been undone."
    )


def test_v8_foot_tokens_land_on_foot_holds(v8_generator):
    """Every emitted foot move must target a foot-eligible hold (role_id == 15).

    The decode-time mask in _build_valid_token_mask already enforces this, but
    this test makes the contract explicit so future mask changes are caught.
    """
    rng = random.Random(42)
    cids = sorted(v8_generator.climb_data.keys())
    sample = rng.sample(cids, k=min(6, len(cids)))

    violations = []
    for cid in sample:
        climb_info = v8_generator.climb_data.get(cid, {})
        hold_role = {h['hole_id']: h.get('role_id') for h in climb_info.get('holds', [])}
        try:
            outs = v8_generator.generate(climb_id=cid, num_sequences=1,
                                         temperature=0.8, max_length=30)
        except Exception:
            continue
        for result in (outs or []):
            for move in (result.get('sequence') or []):
                if move.get('limb') in ('RF', 'LF'):
                    hid = move.get('hold')
                    role = hold_role.get(hid)
                    if role is not None and role != 15:
                        violations.append(
                            f"climb {cid}: {move['limb']} on hold {hid} (role_id={role})"
                        )

    assert not violations, (
        f"Foot moves landed on non-foot holds:\n" + "\n".join(violations)
    )


def test_v8_emits_named_transitions(v8_generator):
    """At least some generated sequences should carry transition_type metadata
    (present in v9+ beam export; may be absent in legacy v8 data — test is
    advisory, not a hard failure)."""
    rng = random.Random(7)
    cids = sorted(v8_generator.climb_data.keys())
    sample = rng.sample(cids, k=min(6, len(cids)))

    found_ttypes: set = set()
    for cid in sample:
        try:
            outs = v8_generator.generate(climb_id=cid, num_sequences=1,
                                         temperature=0.8, max_length=30)
        except Exception:
            continue
        for result in (outs or []):
            for move in (result.get('sequence') or []):
                tt = move.get('transition_type')
                if tt:
                    found_ttypes.add(tt)

    # Advisory: warn but don't hard-fail if v8 data predates transition_type.
    if not found_ttypes:
        pytest.skip(
            "No transition_type found in generated sequences — expected for pre-v9 "
            "models. Re-run against v9 model after data re-export."
        )
    foot_ttypes = {'step_up', 'back_step', 'high_step', 'match',
                   'replant', 'unflag', 'foot_move', 'start_foot'}
    assert found_ttypes & foot_ttypes, (
        f"transition_type present but no foot-specific types found. Got: {found_ttypes}"
    )
