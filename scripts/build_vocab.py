#!/usr/bin/env python3
import json
import argparse
import os

def build_vocab(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)

    hold_mapping = {'PAD': 0}
    reverse = {0: 'PAD'}
    counter = 1

    for item in data.get('results', []):
        holds = item.get('holds', []) or item.get('best_sequence', {}).get('holds', [])
        for h in holds:
            hid = h.get('hole_id')
            if hid is None:
                continue
            if hid not in hold_mapping:
                hold_mapping[hid] = counter
                reverse[counter] = hid
                counter += 1

    payload = {
        'hold_mapping': {str(k): v for k, v in hold_mapping.items()},
        'reverse_hold_mapping': {str(k): v for k, v in reverse.items()},
        'limb_mapping': {'RH': 0, 'LH': 1, 'RF': 2, 'LF': 3},
        'role_mapping': {"12": "Start", "13": "Hand", "14": "Finish", "15": "Foot"}
    }

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote vocab with {len(hold_mapping)} holds to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Build a hold vocab from climbs JSON')
    parser.add_argument('--input', required=True, help='Input climbs JSON (exported training or raw)')
    parser.add_argument('--output', required=True, help='Output vocab JSON file')
    args = parser.parse_args()
    build_vocab(args.input, args.output)


if __name__ == '__main__':
    main()
