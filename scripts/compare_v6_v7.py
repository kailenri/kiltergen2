"""Side-by-side qualitative comparison of v6 vs v7 LSTM generations.

For each of N overlapping climbs:
  - Generate one sequence with v6 model and one with v7 model
  - Score both via ClimbSequenceGenerator.evaluate_sequence
  - Render a 2-up viz with matplotlib

The v6 checkpoint predates the (quality, dir_x, dir_y) feature triple, so
its hold_encoder expects 6 hold features instead of the current 9. We
rebuild a v6-compatible model in-place by swapping the `hold_features`
buffer and `hold_encoder` submodule before load_state_dict.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lstm import ClimbGenerator, ClimbLSTM, HoldEncoder, NUM_DIFFICULTY_BUCKETS
from sequence_generator import ClimbSequenceGenerator
from viz import plot_climb_sequence


def patch_for_v6(gen: ClimbGenerator, v6_ckpt: str) -> bool:
    """Build a v6-compatible model on top of `gen.dataset` and load weights."""
    spatial_13 = gen.dataset.build_spatial_features()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = ClimbLSTM(
        vocab_size=gen.dataset.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        spatial_features=spatial_13,
        num_difficulty_buckets=NUM_DIFFICULTY_BUCKETS,
        num_holds=len(gen.dataset.hold_mapping),
    )

    # v6 used 6 hold features: [x_norm, y_norm, role_oh×4]; replace the
    # current 9-feature buffer + hold_encoder so checkpoint shapes match.
    del model.hold_features
    model.register_buffer("hold_features", spatial_13[:, :6].clone())
    model.hold_encoder = HoldEncoder(
        hold_feat_dim=6, hold_embed_dim=32, num_limbs=4, limb_embed_dim=8
    )

    state = torch.load(v6_ckpt, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        print(f"  v6 state_dict load failed: {exc}")
        return False
    model.to(device).eval()
    gen.model = model
    print(f"  v6 model loaded (compat shim): {v6_ckpt}")
    return True


def gen_one(gen: ClimbGenerator, climb_id, temperature: float, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    out = gen.generate(
        climb_id=climb_id, num_sequences=1, temperature=temperature, max_length=30
    )
    return out[0] if out else None


def score_details(holds, sequence):
    """Return realism subscores from ClimbSequenceGenerator.evaluate_sequence."""
    if not holds or not sequence:
        return {}
    sg = ClimbSequenceGenerator(holds)
    ev = sg.evaluate_sequence(sequence)
    d = ev.get("details", {})
    return {
        "score": ev.get("score", 0.0),
        "hold_quality": d.get("hold_quality", 0.0),
        "hold_alignment": d.get("hold_alignment", 0.0),
        "flag_quality": d.get("flag_quality", 0.0),
        "cross_prevention": d.get("cross_prevention", 0.0),
        "body_position": d.get("body_position", 0.0),
        "movement_efficiency": d.get("movement_efficiency", 0.0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v6-data", default="data/training_sequences_subset_5k_v6.json")
    p.add_argument("--v6-vocab", default="data/vocab_subset_5k_v6.json")
    p.add_argument("--v6-model", default="models/subset_v6_model.pth")
    p.add_argument("--v7-data", default="data/training_sequences_subset_5k_v7.json")
    p.add_argument("--v7-vocab", default="data/vocab_subset_5k_v7.json")
    p.add_argument("--v7-model", default="models/subset_v7_model.pth")
    p.add_argument("--n-climbs", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="comparisons/v6_vs_v7")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading v6 generator: {args.v6_data}")
    g6 = ClimbGenerator(args.v6_data, vocab_path=args.v6_vocab)
    if not patch_for_v6(g6, args.v6_model):
        print("Abort: could not load v6 model.")
        return

    print(f"\nLoading v7 generator: {args.v7_data}")
    g7 = ClimbGenerator(args.v7_data, vocab_path=args.v7_vocab)
    if not g7.load_model(args.v7_model):
        print("Abort: could not load v7 model.")
        return

    overlap = sorted(set(g6.climb_data.keys()) & set(g7.climb_data.keys()))
    print(f"\nOverlapping climbs: {len(overlap)}")
    if not overlap:
        print("No overlapping climbs.")
        return

    rng = random.Random(args.seed)
    picks = rng.sample(overlap, k=min(args.n_climbs, len(overlap)))

    summary_rows = []

    for idx, cid in enumerate(picks, 1):
        name = g7.climb_data[cid]["name"]
        print(f"\n[{idx}/{len(picks)}] climb_id={cid} name={name!r}")

        s6 = gen_one(g6, cid, args.temperature, args.seed + idx)
        s7 = gen_one(g7, cid, args.temperature, args.seed + idx)
        if not s6 or not s7:
            print("  skipped: generation failed")
            continue

        seq6, seq7 = s6["sequence"], s7["sequence"]
        # Both datasets store the same canonical holds list per climb;
        # use v7 holds for scoring (orientations data matches v7 export).
        holds = g7.climb_data[cid]["holds"]
        m6 = score_details(holds, seq6)
        m7 = score_details(holds, seq7)

        print(
            f"  v6: len={len(seq6):2d}  score={m6.get('score',0):.2f}  "
            f"align={m6.get('hold_alignment',0):.2f}  "
            f"flag={m6.get('flag_quality',0):.2f}  "
            f"cross={m6.get('cross_prevention',0):.2f}"
        )
        print(
            f"  v7: len={len(seq7):2d}  score={m7.get('score',0):.2f}  "
            f"align={m7.get('hold_alignment',0):.2f}  "
            f"flag={m7.get('flag_quality',0):.2f}  "
            f"cross={m7.get('cross_prevention',0):.2f}"
        )

        safe_name = "".join(c if c.isalnum() else "_" for c in name)[:40]
        base = f"{idx:02d}_{cid}_{safe_name}"

        path6 = os.path.join(args.output_dir, f"{base}__v6.png")
        path7 = os.path.join(args.output_dir, f"{base}__v7.png")
        title6 = (
            f"v6  {name}  | score={m6.get('score',0):.2f} "
            f"flag={m6.get('flag_quality',0):.2f} align={m6.get('hold_alignment',0):.2f}"
        )
        title7 = (
            f"v7  {name}  | score={m7.get('score',0):.2f} "
            f"flag={m7.get('flag_quality',0):.2f} align={m7.get('hold_alignment',0):.2f}"
        )
        try:
            plot_climb_sequence(holds, seq6, title=title6, output_path=path6, show=False)
            plot_climb_sequence(holds, seq7, title=title7, output_path=path7, show=False)
        except Exception as exc:
            print(f"  viz error: {exc}")
            continue

        summary_rows.append({
            "climb_id": cid,
            "name": name,
            "v6": m6,
            "v7": m7,
            "v6_len": len(seq6),
            "v7_len": len(seq7),
        })

    # Aggregate summary
    if summary_rows:
        keys = ["score", "hold_quality", "hold_alignment", "flag_quality",
                "cross_prevention", "body_position", "movement_efficiency"]
        print("\n" + "=" * 72)
        print("AGGREGATE (mean across climbs)")
        print("=" * 72)
        print(f"{'metric':22s}  {'v6':>10s}  {'v7':>10s}  {'Δ (v7-v6)':>12s}")
        for k in keys:
            v6m = sum(r["v6"].get(k, 0.0) for r in summary_rows) / len(summary_rows)
            v7m = sum(r["v7"].get(k, 0.0) for r in summary_rows) / len(summary_rows)
            print(f"{k:22s}  {v6m:10.3f}  {v7m:10.3f}  {v7m - v6m:+12.3f}")
        v6len = sum(r["v6_len"] for r in summary_rows) / len(summary_rows)
        v7len = sum(r["v7_len"] for r in summary_rows) / len(summary_rows)
        print(f"{'seq_length':22s}  {v6len:10.2f}  {v7len:10.2f}  {v7len - v6len:+12.2f}")

        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"\nWrote {len(summary_rows)} comparison(s) → {args.output_dir}/")
        print(f"Summary JSON → {summary_path}")


if __name__ == "__main__":
    main()
