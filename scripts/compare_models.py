"""N-way model comparison with foot-aware realism metrics.

For each of N overlapping climbs, generate one sequence per model with a
fixed seed, score with `ClimbSequenceGenerator.evaluate_sequence`, count
foot moves / limb distribution, and render side-by-side viz.

Models are specified one per `--model` flag:
    --model LABEL=MODEL_PATH,VOCAB_PATH,DATA_PATH[,FLAGS]

`FLAGS` is comma-separated; the only currently-meaningful flag is
`v6compat` which applies the 6-hold-feature shim for pre-Phase-F models.

Example:
    python scripts/compare_models.py \\
        --model v6=models/subset_v6_model.pth,data/vocab_subset_5k_v6.json,data/training_sequences_subset_5k_v6.json,v6compat \\
        --model v7=models/subset_v7_model.pth,data/vocab_subset_5k_v7.json,data/training_sequences_subset_5k_v7.json \\
        --n-climbs 6 --output-dir comparisons/v6_v7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lstm import ClimbGenerator, ClimbLSTM, HoldEncoder, NUM_DIFFICULTY_BUCKETS
from sequence_generator import ClimbSequenceGenerator
from viz import plot_climb_sequence, plot_climb_sequence_animated


@dataclass
class ModelSpec:
    label: str
    model_path: str
    vocab_path: str
    data_path: str
    flags: List[str] = field(default_factory=list)

    @classmethod
    def parse(cls, s: str) -> "ModelSpec":
        if "=" not in s:
            raise ValueError(f"--model expects LABEL=paths..., got: {s!r}")
        label, rest = s.split("=", 1)
        parts = rest.split(",")
        if len(parts) < 3:
            raise ValueError(f"--model needs MODEL,VOCAB,DATA[,FLAGS], got: {s!r}")
        return cls(
            label=label,
            model_path=parts[0],
            vocab_path=parts[1],
            data_path=parts[2],
            flags=[p for p in parts[3:] if p],
        )


def patch_for_v6(gen: ClimbGenerator, ckpt: str) -> bool:
    """v6 used 6 hold features (no quality/dir); rebuild matching submodules."""
    spatial_13 = gen.dataset.build_spatial_features()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = ClimbLSTM(
        vocab_size=gen.dataset.vocab_size,
        embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2,
        spatial_features=spatial_13,
        num_difficulty_buckets=NUM_DIFFICULTY_BUCKETS,
        num_holds=len(gen.dataset.hold_mapping),
    )
    del model.hold_features
    model.register_buffer("hold_features", spatial_13[:, :6].clone())
    model.hold_encoder = HoldEncoder(hold_feat_dim=6, hold_embed_dim=32,
                                     num_limbs=4, limb_embed_dim=8)
    try:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    except RuntimeError as exc:
        print(f"  v6 state_dict load failed: {exc}")
        return False
    model.to(device).eval()
    gen.model = model
    return True


def load_generator(spec: ModelSpec) -> Optional[ClimbGenerator]:
    print(f"\n=== Loading [{spec.label}] {spec.model_path}")
    gen = ClimbGenerator(spec.data_path, vocab_path=spec.vocab_path)
    if "v6compat" in spec.flags:
        ok = patch_for_v6(gen, spec.model_path)
    else:
        ok = gen.load_model(spec.model_path)
    if not ok:
        print(f"  [{spec.label}] load FAILED")
        return None
    return gen


def gen_one(gen: ClimbGenerator, climb_id, temperature: float, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    out = gen.generate(climb_id=climb_id, num_sequences=1,
                       temperature=temperature, max_length=30)
    return out[0] if out else None


def metrics_for(holds, sequence) -> Dict:
    """Realism subscores + foot stats for a generated sequence."""
    if not holds or not sequence:
        return {}
    sg = ClimbSequenceGenerator(holds)
    ev = sg.evaluate_sequence(sequence)
    d = ev.get("details", {})

    limb_counts = Counter(m.get("limb") for m in sequence)
    n_foot = limb_counts["RF"] + limb_counts["LF"]
    n_hand = limb_counts["RH"] + limb_counts["LH"]
    n_total = max(1, len(sequence))

    finish_ids = {h["hole_id"] for h in holds if h.get("role_id") == 14}
    reached = bool(finish_ids) and any(m["hold"] in finish_ids for m in sequence[-2:])

    return {
        "len": len(sequence),
        "score": ev.get("score", 0.0),
        "hold_quality": d.get("hold_quality", 0.0),
        "hold_alignment": d.get("hold_alignment", 0.0),
        "flag_quality": d.get("flag_quality", 0.0),
        "cross_prevention": d.get("cross_prevention", 0.0),
        "body_position": d.get("body_position", 0.0),
        "movement_efficiency": d.get("movement_efficiency", 0.0),
        "n_foot": n_foot,
        "n_hand": n_hand,
        "foot_pct": n_foot / n_total,
        "finish_reached": reached,
        "RH": limb_counts["RH"], "LH": limb_counts["LH"],
        "RF": limb_counts["RF"], "LF": limb_counts["LF"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help="LABEL=MODEL,VOCAB,DATA[,FLAGS]")
    p.add_argument("--n-climbs", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="comparisons/multi")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip PNG rendering (faster for stats-only runs)")
    p.add_argument("--animate", action="store_true",
                   help="Also emit a per-model GIF showing the stick figure stepping through the climb")
    p.add_argument("--animate-fps", type=float, default=1.2,
                   help="Frames per second for --animate output (default 1.2)")
    args = p.parse_args()

    specs = [ModelSpec.parse(s) for s in args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    gens: Dict[str, ClimbGenerator] = {}
    for spec in specs:
        g = load_generator(spec)
        if g is None:
            print(f"Abort: failed to load {spec.label}")
            return
        gens[spec.label] = g

    # Climb-id intersection across all loaded generators.
    overlap = None
    for label, g in gens.items():
        ids = set(g.climb_data.keys())
        overlap = ids if overlap is None else (overlap & ids)
    overlap = sorted(overlap or [])
    print(f"\nOverlapping climbs across {len(gens)} model(s): {len(overlap)}")
    if not overlap:
        print("No overlap. Aborting.")
        return

    picks = random.Random(args.seed).sample(overlap, k=min(args.n_climbs, len(overlap)))

    per_climb_rows = []  # rows: dict(climb_id, name, {label: metrics})

    for idx, cid in enumerate(picks, 1):
        # Holds: prefer the first generator that has the climb (they should all match
        # since these are the same canonical climb data, even if augmentation differs).
        any_gen = next(iter(gens.values()))
        name = any_gen.climb_data[cid]["name"]
        holds = any_gen.climb_data[cid]["holds"]
        print(f"\n[{idx}/{len(picks)}] {cid}  {name!r}")

        row = {"climb_id": cid, "name": name, "by_model": {}}

        for label, g in gens.items():
            s = gen_one(g, cid, args.temperature, args.seed + idx)
            if not s:
                print(f"  [{label}] generation failed")
                row["by_model"][label] = None
                continue
            seq = s["sequence"]
            if not seq:
                print(f"  [{label:6s}] empty sequence (skipped)")
                row["by_model"][label] = None
                continue
            m = metrics_for(holds, seq)
            row["by_model"][label] = {"metrics": m, "sequence": seq}
            print(
                f"  [{label:6s}] len={m['len']:2d} feet={m['n_foot']} "
                f"score={m['score']:.2f} flag={m['flag_quality']:.2f} "
                f"align={m['hold_alignment']:.2f} cross={m['cross_prevention']:.2f} "
                f"finish={'Y' if m['finish_reached'] else 'N'}"
            )

            if not args.no_viz:
                safe = "".join(c if c.isalnum() else "_" for c in name)[:40]
                base = f"{idx:02d}_{cid}_{safe}__{label}.png"
                title = (
                    f"[{label}] {name}  | score={m['score']:.2f} "
                    f"flag={m['flag_quality']:.2f} feet={m['n_foot']}/{m['len']}"
                )
                try:
                    plot_climb_sequence(holds, seq, title=title,
                                        output_path=os.path.join(args.output_dir, base),
                                        show=False)
                except Exception as exc:
                    print(f"  [{label}] viz error: {exc}")
                if args.animate:
                    gif_base = f"{idx:02d}_{cid}_{safe}__{label}.gif"
                    try:
                        plot_climb_sequence_animated(
                            holds, seq, title=title,
                            output_path=os.path.join(args.output_dir, gif_base),
                            fps=args.animate_fps,
                        )
                    except Exception as exc:
                        print(f"  [{label}] animation error: {exc}")

        per_climb_rows.append(row)

    # ---------- Aggregate table ----------
    labels = [s.label for s in specs]
    print("\n" + "=" * 88)
    print(f"AGGREGATE over {len(per_climb_rows)} climbs (mean unless noted)")
    print("=" * 88)
    header = f"{'metric':22s}" + "".join(f"  {l:>11s}" for l in labels)
    print(header)
    print("-" * len(header))

    metric_keys = [
        "len", "n_foot", "foot_pct",
        "score", "hold_quality", "hold_alignment",
        "flag_quality", "cross_prevention", "body_position",
        "movement_efficiency",
    ]
    for k in metric_keys:
        means = []
        for l in labels:
            vals = [r["by_model"][l]["metrics"][k] for r in per_climb_rows
                    if r["by_model"].get(l)]
            means.append(sum(vals) / len(vals) if vals else 0.0)
        line = f"{k:22s}"
        for v in means:
            line += f"  {v:>11.3f}"
        print(line)

    # Finish-reached %
    line = f"{'finish_reached_pct':22s}"
    for l in labels:
        flags = [int(r["by_model"][l]["metrics"]["finish_reached"])
                 for r in per_climb_rows if r["by_model"].get(l)]
        line += f"  {(100*sum(flags)/max(1,len(flags))):>10.1f}%"
    print(line)

    # Median feet
    line = f"{'median feet/seq':22s}"
    for l in labels:
        vs = sorted(r["by_model"][l]["metrics"]["n_foot"] for r in per_climb_rows
                    if r["by_model"].get(l))
        med = vs[len(vs)//2] if vs else 0
        line += f"  {med:>11.1f}"
    print(line)

    # ---------- Stop-criterion check ----------
    print("\n" + "=" * 88)
    print("STOP-CRITERION CHECK  (target: median feet ≥ 2  AND  mean flag_quality > 0.3)")
    print("=" * 88)
    for l in labels:
        ms = [r["by_model"][l]["metrics"] for r in per_climb_rows
              if r["by_model"].get(l)]
        if not ms:
            continue
        med_feet = sorted(m["n_foot"] for m in ms)[len(ms)//2]
        mean_flag = sum(m["flag_quality"] for m in ms) / len(ms)
        ok = (med_feet >= 2) and (mean_flag > 0.3)
        print(f"  [{l:6s}] median feet={med_feet}  mean flag_quality={mean_flag:.3f}  → "
              f"{'PASS' if ok else 'FAIL'}")

    # ---------- Save JSON ----------
    out_json = os.path.join(args.output_dir, "summary.json")
    # Drop sequence from saved JSON (verbose); keep metrics.
    save_rows = [
        {
            "climb_id": r["climb_id"],
            "name": r["name"],
            "by_model": {l: (e["metrics"] if e else None)
                         for l, e in r["by_model"].items()},
        }
        for r in per_climb_rows
    ]
    with open(out_json, "w") as f:
        json.dump(save_rows, f, indent=2)
    print(f"\nSummary JSON: {out_json}")
    if not args.no_viz:
        print(f"Viz PNGs   : {args.output_dir}/")


if __name__ == "__main__":
    main()
