"""Analyze foot-token prevalence in a training sequences JSON file.

Quick distribution dump answering:
  - What % of sequences contain at least one foot move?
  - What % of all tokens are RF / LF?
  - How many foot moves does a typical sequence have?
  - What's the distribution of sequence lengths?

Usage:
    python scripts/analyze_data_distribution.py --data data/training_sequences_subset_5k_v7.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/training_sequences_subset_5k_v7.json")
    args = p.parse_args()

    print(f"Loading {args.data}")
    with open(args.data) as f:
        d = json.load(f)

    seq_total = 0
    all_tokens = 0
    limb_tok = Counter()
    seq_with_any_foot = 0
    feet_per_seq = []
    seq_lens = []

    for r in d["results"]:
        seq = r.get("best_sequence", {}).get("sequence", [])
        if not seq:
            continue
        seq_total += 1
        seq_lens.append(len(seq))
        foot_count = 0
        for m in seq:
            l = m.get("limb")
            limb_tok[l] += 1
            all_tokens += 1
            if l in ("RF", "LF"):
                foot_count += 1
        feet_per_seq.append(foot_count)
        if foot_count > 0:
            seq_with_any_foot += 1

    print(f"\n== Sequences ==")
    print(f"total                       : {seq_total}")
    print(f"avg length                  : {sum(seq_lens)/seq_total:.2f}")
    print(f"min/median/max length       : {min(seq_lens)}/{sorted(seq_lens)[seq_total//2]}/{max(seq_lens)}")
    print(f"contain >= 1 foot move      : {seq_with_any_foot} ({100*seq_with_any_foot/seq_total:.1f}%)")
    avg_feet = sum(feet_per_seq) / seq_total
    feet_per_seq_sorted = sorted(feet_per_seq)
    median_feet = feet_per_seq_sorted[seq_total//2]
    print(f"feet per sequence avg/median: {avg_feet:.2f} / {median_feet}")

    print(f"\n== Tokens ==")
    print(f"total                       : {all_tokens}")
    for l in ("RH", "LH", "RF", "LF"):
        c = limb_tok[l]
        print(f"  {l:3s}                       : {c:>7d}  ({100*c/all_tokens:.1f}%)")
    foot_pct = 100 * (limb_tok["RF"] + limb_tok["LF"]) / all_tokens
    hand_pct = 100 * (limb_tok["RH"] + limb_tok["LH"]) / all_tokens
    print(f"  feet (RF+LF)              : {limb_tok['RF']+limb_tok['LF']:>7d}  ({foot_pct:.1f}%)")
    print(f"  hands (RH+LH)             : {limb_tok['RH']+limb_tok['LH']:>7d}  ({hand_pct:.1f}%)")

    print(f"\n== Histogram of foot count per sequence ==")
    feet_hist = Counter(feet_per_seq)
    for k in sorted(feet_hist):
        bar = "#" * min(60, feet_hist[k] // max(1, seq_total // 100))
        print(f"  {k:2d} feet : {feet_hist[k]:>6d}  {bar}")


if __name__ == "__main__":
    main()
