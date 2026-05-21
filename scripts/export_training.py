#!/usr/bin/env python3
import json
import os
import sys
import argparse
from tqdm import tqdm

# allow running from scripts/ so imports find top-level modules
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sequence_generator import ClimbSequenceGenerator


def _sequence_is_reachable(gen, seq):
    limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}

    for move in seq:
        limb = move.get('limb')
        hold_id = move.get('hold')
        if limb not in limb_positions or hold_id not in gen.hold_dict:
            return False

        current_hold = limb_positions[limb]
        current_id = current_hold['hole_id'] if current_hold else -1
        current_limbs_tuple = tuple(sorted(
            (l, h['hole_id'] if h else -1)
            for l, h in limb_positions.items()
        ))

        if limb in ('RH', 'LH'):
            if not gen.is_valid_hand_transition(current_id, hold_id, limb, current_limbs_tuple):
                return False
        else:
            if not gen.is_valid_foot_transition(current_id, hold_id, limb, current_limbs_tuple):
                return False

        limb_positions[limb] = gen.hold_dict[hold_id]

    return True


def export_training(input_file, output_file, per_climb=5, beam_width=6):
    with open(input_file) as f:
        data = json.load(f)

    results = []
    for result in tqdm(data.get('results', []), desc='Climbs'):
        climb_id = result.get('id')
        name = result.get('name')
        holds = result.get('best_sequence', {}).get('holds', [])
        if not holds:
            continue

        try:
            gen = ClimbSequenceGenerator(holds)
        except Exception as e:
            print(f"Skipping climb {climb_id}: init error: {e}")
            continue

        res = gen.generate_sequences(beam_width=beam_width)
        if res.get('status') != 'success':
            print(f"No sequences for climb {climb_id}")
            continue

        def _seq_is_valid(seq):
            """Quality filter: ends on finish hold, hands on hand holds, reachable."""
            if not seq:
                return False
            if gen.hold_dict.get(seq[-1]['hold'], {}).get('role_id') != 14:
                return False
            for m in seq:
                if m['limb'] in ('RH', 'LH'):
                    if gen.hold_dict.get(m['hold'], {}).get('role_id') not in {12, 13, 14}:
                        return False
            return _sequence_is_reachable(gen, seq)

        all_seqs = res.get('all_sequences', [])
        # Separate sequences with/without foot moves so we target a balanced mix.
        regular_seqs = [s for s in all_seqs
                        if not any(m['limb'] in ('RF', 'LF') for m in s.get('sequence', []))]
        foot_seqs    = [s for s in all_seqs
                        if     any(m['limb'] in ('RF', 'LF') for m in s.get('sequence', []))]

        n_regular_target = max(1, (per_climb + 1) // 2)   # ceil(per_climb / 2)
        n_foot_target    = per_climb // 2

        n_regular_kept = n_foot_kept = 0
        seq_i = 0
        for seqobj in regular_seqs + foot_seqs:
            if n_regular_kept + n_foot_kept >= per_climb:
                break
            seq = seqobj.get('sequence', [])
            if not _seq_is_valid(seq):
                continue
            has_foot = any(m['limb'] in ('RF', 'LF') for m in seq)
            if has_foot:
                if n_foot_kept >= n_foot_target:
                    continue
                n_foot_kept += 1
            else:
                if n_regular_kept >= n_regular_target:
                    continue
                n_regular_kept += 1
            results.append({
                'id': f"{climb_id}_gen_{seq_i}",
                'name': name,
                'difficulty': result.get('difficulty'),
                'best_sequence': {
                    'holds': holds,
                    'sequence': seq
                },
                'evaluation': gen.evaluate_sequence(seq)
            })
            seq_i += 1

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Export cleaned training sequences using the beam generator')
    parser.add_argument('--input', required=True, help='Input climbs JSON file')
    parser.add_argument('--output', required=True, help='Output training JSON file')
    parser.add_argument('--per-climb', type=int, default=5, help='Max sequences per climb to keep')
    parser.add_argument('--beam-width', type=int, default=6, help='Beam width to use for generation')

    args = parser.parse_args()
    export_training(args.input, args.output, per_climb=args.per_climb, beam_width=args.beam_width)


if __name__ == '__main__':
    main()
