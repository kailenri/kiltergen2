#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm

from sequence_generator import ClimbSequenceGenerator


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
        for seqobj in res.get('all_sequences', [])[:per_climb]:
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

            evaluation = gen.evaluate_sequence(seq)
            cleaned.append({'sequence': seq, 'evaluation': evaluation})

        if cleaned:
            results.append({'climb_id': climb_id, 'name': name, 'holds': holds, 'sequences': cleaned})

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
