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
