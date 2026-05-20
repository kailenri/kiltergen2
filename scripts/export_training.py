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

        cleaned = []
        for seq_i, seqobj in enumerate(res.get('all_sequences', [])[:per_climb]):
            seq = seqobj.get('sequence', [])
            if not seq:
                continue

            # require final move to be a finish hold
            last = seq[-1]
            last_role = gen.hold_dict.get(last['hold'], {}).get('role_id')
            if last_role != 14:
                continue

            # ensure hands are only on hand holds
            ok = True
            for m in seq:
                if m['limb'] in ('RH', 'LH'):
                    role = gen.hold_dict.get(m['hold'], {}).get('role_id')
                    if role not in {12, 13, 14}:
                        ok = False
                        break
            if not ok:
                continue

            # ensure reachability matches generator rules
            if not _sequence_is_reachable(gen, seq):
                continue

            evaluation = gen.evaluate_sequence(seq)
            # create a result entry per sequence so ClimbDataset can consume it
            results.append({
                'id': f"{climb_id}_gen_{seq_i}",
                'name': name,
                'best_sequence': {
                    'holds': holds,
                    'sequence': seq
                },
                'evaluation': evaluation
            })

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
