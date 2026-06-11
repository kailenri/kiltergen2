#!/usr/bin/env python3
"""Build per-hold orientation + quality from board image crops.

For each unique hold (hole_id) in a climbs JSON, crops a region from the
blank Kilter board image at the hold's calibrated pixel position, runs
Sobel edge detection, and writes per-hold:
  - direction_deg: dominant edge-tangent direction in [-180, 180)
    (the orientation of the hold's main 'lip' / surface line)
  - quality: heuristic in [0.4, 1.0] from edge density
  - edge_density: raw mean |gradient|, normalized to [0, 1]
  - confidence: a heuristic combining edge density and gradient
    coherence; low when the crop is uniform or chaotic.

Output: data/hold_orientations.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from viz import _hold_to_px, _load_board_image  # noqa: E402


def _sobel_dxdy(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Plain 3x3 Sobel, no scipy dependency."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    pad = np.pad(gray, 1, mode='edge')
    out_x = np.zeros_like(gray, dtype=np.float32)
    out_y = np.zeros_like(gray, dtype=np.float32)
    for j in range(3):
        for i in range(3):
            patch = pad[j:j + gray.shape[0], i:i + gray.shape[1]]
            out_x += kx[j, i] * patch
            out_y += ky[j, i] * patch
    return out_x, out_y


def analyze_crop(crop_rgb: np.ndarray) -> Dict[str, float]:
    """Return {direction_deg, quality, edge_density, confidence} for one crop."""
    if crop_rgb.size == 0:
        return {'direction_deg': 0.0, 'quality': 0.7,
                'edge_density': 0.0, 'confidence': 0.0}
    if crop_rgb.ndim == 3:
        gray = crop_rgb[..., :3].mean(axis=-1).astype(np.float32)
    else:
        gray = crop_rgb.astype(np.float32)
    # Normalize to [0, 1] roughly
    g = gray / (gray.max() + 1e-6)

    gx, gy = _sobel_dxdy(g)
    mag = np.hypot(gx, gy)
    mag_mean = float(mag.mean())
    # Normalize edge density: Sobel on [0,1] image typically peaks ~4-6.
    edge_density = float(min(1.0, mag_mean / 0.4))

    # Dominant orientation via doubled-angle averaging (so opposite gradients
    # don't cancel). Theta = atan2(2*gx*gy, gx^2 - gy^2) / 2.
    weights = mag  # weight by gradient magnitude
    sxx = float(((gx * gx - gy * gy) * weights).sum())
    sxy = float((2.0 * gx * gy * weights).sum())
    if abs(sxx) + abs(sxy) < 1e-6:
        direction_deg = 0.0
        coherence = 0.0
    else:
        theta = 0.5 * math.atan2(sxy, sxx)
        # Convert from gradient axis to edge-tangent direction (perpendicular).
        edge_tangent = theta + math.pi / 2.0
        direction_deg = math.degrees(edge_tangent)
        # Wrap to [-180, 180).
        direction_deg = ((direction_deg + 180.0) % 360.0) - 180.0
        coherence = math.hypot(sxx, sxy) / (float(weights.sum() * (g.size ** 0.5)) + 1e-6)
        coherence = float(min(1.0, coherence * 4.0))

    # Quality: high edge density = good (positive features the fingers can
    # catch). Map [0, 1] edge density to [0.5, 1.0].
    quality = 0.5 + 0.5 * edge_density
    # Confidence: needs both signal and coherence to be high.
    confidence = float(min(1.0, 0.5 * edge_density + 0.5 * coherence))

    return {
        'direction_deg': float(round(direction_deg, 2)),
        'quality': float(round(quality, 3)),
        'edge_density': float(round(edge_density, 3)),
        'confidence': float(round(confidence, 3)),
    }


def _iter_holds(input_path: str):
    """Yield unique (hole_id, hold_dict) pairs from a climbs JSON."""
    with open(input_path) as f:
        data = json.load(f)
    seen = set()
    for result in data.get('results', []):
        holds = (result.get('holds')
                 or result.get('best_sequence', {}).get('holds')
                 or [])
        for h in holds:
            hid = h.get('hole_id')
            if hid is None or hid in seen:
                continue
            seen.add(hid)
            yield hid, h


def build_orientations(input_path: str, output_path: str,
                       crop_radius: int = 16) -> Dict[int, Dict]:
    img, w, h = _load_board_image()
    if img is None:
        raise RuntimeError("Could not load board image (see config in viz.py)")

    img_arr = np.asarray(img)
    results: Dict[int, Dict] = {}
    skipped = 0
    for hid, hold in _iter_holds(input_path):
        px, py, _r = _hold_to_px(hold)
        cx, cy = int(round(px)), int(round(py))
        x0 = max(0, cx - crop_radius)
        x1 = min(img_arr.shape[1], cx + crop_radius)
        y0 = max(0, cy - crop_radius)
        y1 = min(img_arr.shape[0], cy + crop_radius)
        if x1 - x0 < 4 or y1 - y0 < 4:
            skipped += 1
            continue
        crop = img_arr[y0:y1, x0:x1]
        info = analyze_crop(crop)
        info['px'] = int(cx)
        info['py'] = int(cy)
        results[hid] = info

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        'meta': {
            'source_image': 'base_kilter_image/blank_kilter.jpeg',
            'crop_radius': crop_radius,
            'n_holds': len(results),
            'n_skipped': skipped,
        },
        'holds': {str(hid): info for hid, info in results.items()},
    }
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                        help='Climbs JSON to source unique hole_ids from')
    parser.add_argument('--output', default='data/hold_orientations.json')
    parser.add_argument('--crop-radius', type=int, default=16)
    args = parser.parse_args()
    results = build_orientations(args.input, args.output, args.crop_radius)
    print(f"Wrote {len(results)} hold orientations to {args.output}")


if __name__ == '__main__':
    main()
