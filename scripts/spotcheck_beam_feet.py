"""Sanity check: confirm Phase A beam now emits foot moves.

Runs ClimbSequenceGenerator on the first N climbs of a climbs json,
prints per-climb stats and an aggregate.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence_generator import ClimbSequenceGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--climbs', default='data/climbs_subset_5k.json')
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--beam-width', type=int, default=6)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    with open(args.climbs) as f:
        data = json.load(f)

    rows = data.get('results') or data
    total_seqs = 0
    seqs_with_feet = 0
    all_foot = 0
    all_tok = 0

    for i, c in enumerate(rows[:args.n]):
        holds = c.get('best_sequence', {}).get('holds') or c.get('holds')
        name = c.get('name', f'climb_{i}')
        if not holds:
            print(f'[{i}] {name!r}: no holds; skipping')
            continue
        try:
            gen = ClimbSequenceGenerator(holds)
            res = gen.generate_sequences(beam_width=args.beam_width)
        except Exception as e:
            print(f'[{i}] {name!r}: error {e}')
            continue
        if res.get('status') != 'success':
            print(f'[{i}] {name!r}: status={res.get("status")}')
            continue
        seqs = res.get('all_sequences', [])
        n_with = 0
        for s in seqs:
            seq = s.get('sequence', [])
            nf = sum(1 for m in seq if m['limb'] in ('RF', 'LF'))
            total_seqs += 1
            all_tok += len(seq)
            all_foot += nf
            if nf > 0:
                seqs_with_feet += 1
                n_with += 1
        print(f'[{i}] {name!r}: {len(seqs)} seqs, {n_with} with feet')
        if args.verbose and seqs:
            for j, s in enumerate(seqs[:2]):
                seq = s.get('sequence', [])
                nf = sum(1 for m in seq if m['limb'] in ('RF', 'LF'))
                first8 = [(m['limb'], m['hold']) for m in seq[:8]]
                print(f'    seq{j}: len={len(seq)} feet={nf} score={s["evaluation"]["score"]:.2f}')
                print(f'      first8: {first8}')

    pct_seqs = 100 * seqs_with_feet / max(1, total_seqs)
    pct_tok = 100 * all_foot / max(1, all_tok)
    print()
    print(f'OVERALL: {seqs_with_feet}/{total_seqs} seqs have feet ({pct_seqs:.1f}%)')
    print(f'         foot tokens {all_foot}/{all_tok} ({pct_tok:.2f}%)')


if __name__ == '__main__':
    main()
