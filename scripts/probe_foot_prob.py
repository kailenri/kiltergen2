"""Probe: does the v7 LSTM assign meaningful probability mass to foot tokens?

Teacher-forces the ground-truth sequence of N random training climbs and at
each position computes P(foot_token | history) vs P(hand_token | history).
This isolates *model knowledge* from *decode-time behavior*: if the model
puts ~zero mass on feet here, no decode hack will save it and we need
loss reweighting (Phase 3). If it puts decent mass but they lose at
sampling time, decode-side fixes (Phase 2) should be enough.

Usage:
    python scripts/probe_foot_prob.py \
        --model models/subset_v7_model.pth \
        --data  data/training_sequences_subset_5k_v7.json \
        --vocab data/vocab_subset_5k_v7.json \
        --n-climbs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lstm import ClimbGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/subset_v7_model.pth")
    p.add_argument("--data", default="data/training_sequences_subset_5k_v7.json")
    p.add_argument("--vocab", default="data/vocab_subset_5k_v7.json")
    p.add_argument("--n-climbs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.data}")
    gen = ClimbGenerator(args.data, vocab_path=args.vocab)
    if not gen.load_model(args.model):
        print("Failed to load model.")
        return

    device = next(gen.model.parameters()).device
    n_limbs = len(gen.dataset.limb_mapping)
    vocab_size = gen.dataset.vocab_size

    # Precompute which token ids are hands vs feet.
    # token = (hold_token * n_limbs) + limb_token + 1
    # limb_token: 0=RH, 1=LH, 2=RF, 3=LF (per dataset.limb_mapping)
    limb_of_token = torch.full((vocab_size,), -1, dtype=torch.long)
    for t in range(1, vocab_size):
        limb_of_token[t] = (t - 1) % n_limbs
    rh, lh, rf, lf = (
        gen.dataset.limb_mapping["RH"],
        gen.dataset.limb_mapping["LH"],
        gen.dataset.limb_mapping["RF"],
        gen.dataset.limb_mapping["LF"],
    )
    is_hand = (limb_of_token == rh) | (limb_of_token == lh)
    is_foot = (limb_of_token == rf) | (limb_of_token == lf)
    is_hand = is_hand.to(device)
    is_foot = is_foot.to(device)

    # Pick climbs whose ground-truth sequence has at least one foot move,
    # so we have positions where the *correct* answer is a foot token.
    climb_ids = list(gen.climb_data.keys())
    random.shuffle(climb_ids)

    chosen = []
    for cid in climb_ids:
        seq = gen.climb_data[cid].get("sequences", [[]])[0]
        if not seq or len(seq) < 4:
            continue
        if any(m.get("limb") in ("RF", "LF") for m in seq):
            chosen.append(cid)
        if len(chosen) >= args.n_climbs:
            break

    if not chosen:
        print("No climbs with feet found in this dataset.")
        return

    print(f"Sampling {len(chosen)} climbs with feet in ground truth.")

    # Aggregates
    sum_p_hand = 0.0
    sum_p_foot = 0.0
    n_positions = 0
    n_foot_targets = 0
    foot_rank_correct_top1 = 0
    foot_rank_correct_top5 = 0

    # Pre-foot / pre-hand position breakdowns
    p_foot_when_target_foot = []
    p_foot_when_target_hand = []

    for cid in chosen:
        seq = gen.climb_data[cid]["sequences"][0]

        tokens = []
        for m in seq:
            ht = gen.dataset.hold_mapping.get(m["hold"])
            lt = gen.dataset.limb_mapping.get(m["limb"])
            if ht is None or lt is None:
                tokens = []
                break
            tokens.append((ht * n_limbs) + lt + 1)
        if len(tokens) < 2:
            continue

        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            output, _ = gen.model(x)
        logits = output[0]                         # (L-1, vocab)
        probs = F.softmax(logits, dim=-1)          # (L-1, vocab)

        targets = torch.tensor(tokens[1:], device=device)  # (L-1,)
        L = probs.shape[0]

        for t in range(L):
            p = probs[t]                            # (vocab,)
            p_h = float(p[is_hand].sum().item())
            p_f = float(p[is_foot].sum().item())
            sum_p_hand += p_h
            sum_p_foot += p_f
            n_positions += 1

            tgt = int(targets[t].item())
            tgt_limb = (tgt - 1) % n_limbs if tgt > 0 else -1
            if tgt_limb in (rf, lf):
                n_foot_targets += 1
                p_foot_when_target_foot.append(p_f)
                # Rank of correct foot token among top-K
                topk = torch.topk(p, 5).indices.tolist()
                if topk and topk[0] == tgt:
                    foot_rank_correct_top1 += 1
                if tgt in topk:
                    foot_rank_correct_top5 += 1
            elif tgt_limb in (rh, lh):
                p_foot_when_target_hand.append(p_f)

    def fmt_pct(x):
        return f"{x*100:.2f}%"

    print("\n== Aggregate over teacher-forced positions ==")
    print(f"positions             : {n_positions}")
    print(f"foot-target positions : {n_foot_targets} ({fmt_pct(n_foot_targets/max(1,n_positions))})")
    print(f"mean P(hand-token)    : {fmt_pct(sum_p_hand/max(1,n_positions))}")
    print(f"mean P(foot-token)    : {fmt_pct(sum_p_foot/max(1,n_positions))}")

    if p_foot_when_target_foot:
        avg = sum(p_foot_when_target_foot) / len(p_foot_when_target_foot)
        print(f"\nWhen ground-truth IS a foot:")
        print(f"  mean P(foot)        : {fmt_pct(avg)}")
        print(f"  top-1 correct       : {foot_rank_correct_top1}/{n_foot_targets} ({fmt_pct(foot_rank_correct_top1/max(1,n_foot_targets))})")
        print(f"  top-5 correct       : {foot_rank_correct_top5}/{n_foot_targets} ({fmt_pct(foot_rank_correct_top5/max(1,n_foot_targets))})")
    if p_foot_when_target_hand:
        avg = sum(p_foot_when_target_hand) / len(p_foot_when_target_hand)
        print(f"\nWhen ground-truth IS a hand:")
        print(f"  mean P(foot)        : {fmt_pct(avg)}")

    print("\nInterpretation:")
    print("  - If mean P(foot) > 0.1 at foot-target positions → model knows feet, decode is the bottleneck (Phase 2).")
    print("  - If mean P(foot) << 0.05 even at foot-target positions → model never learned feet, need loss reweighting (Phase 3).")


if __name__ == "__main__":
    main()
