"""Detect hold centers in `base_kilter_image/blank_kilter.jpeg` and match them
to the holes table of `db.sqlite3` (product_id=1) so visualizations can map
board (x, y) coordinates to exact image pixel positions.

Pipeline:
  1. HoughCircles detects candidate hold centers + radii in the image.
  2. Pull all DB holes for product_id=1.
  3. Seed an affine transform from rough constants (matching viz.py defaults).
  4. Iteratively: project each DB hold into pixel space, snap to nearest
     detected circle within a tolerance, refit affine from matched pairs.
  5. Write `base_kilter_image/hold_pixel_map.json` keyed by hole_id and a
     debug overlay PNG so the result can be inspected visually.

Run:
    python scripts/calibrate_board_image.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(REPO_ROOT, "base_kilter_image", "blank_kilter.jpeg")
DB_PATH = os.path.join(REPO_ROOT, "db.sqlite3")
OUTPUT_JSON = os.path.join(REPO_ROOT, "base_kilter_image", "hold_pixel_map.json")
DEBUG_OVERLAY = os.path.join(REPO_ROOT, "base_kilter_image", "calibration_debug.png")

# Seed affine — close enough that iterative refinement converges. Eyeballed
# from a debug overlay of detected vs projected holds.
SEED = dict(
    origin_x=5.0,
    origin_y=1310.0,
    scale_x=8.1,
    scale_y=7.4,
    flip_y=True,
)

# Holds beyond these board-x bounds are off the main panel (kickboard
# extensions etc.) and don't appear in this image.
BOARD_X_MIN = 4
BOARD_X_MAX = 144
BOARD_Y_MIN = 4
BOARD_Y_MAX = 156


def detect_circles(image_bgr: np.ndarray) -> np.ndarray:
    """Return Nx3 array of (x, y, r) for detected hold centers in pixel space.

    Strategy: each hold has a small dark mounting-screw dot at its center.
    Those dots are the most reliable, geometrically-stable detection target.
    We use SimpleBlobDetector tuned for small dark blobs.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0  # dark blobs
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.minDistBetweenBlobs = 10

    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 120

    params.filterByCircularity = True
    params.minCircularity = 0.35

    params.filterByConvexity = True
    params.minConvexity = 0.6

    params.filterByInertia = True
    params.minInertiaRatio = 0.25

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    if not keypoints:
        raise RuntimeError("SimpleBlobDetector returned no detections.")

    out = np.array([[kp.pt[0], kp.pt[1], kp.size / 2.0] for kp in keypoints],
                   dtype=np.float64)
    return out


def load_db_holes(db_path: str) -> List[Dict]:
    """Return all placed holes on layout_id=1 with their set_id.

    set_id=1  -> large T-nut holds (main grid)
    set_id=20 -> small screw-on / kickboard foot holds (offset positions)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT h.id, h.name, h.x, h.y, p.set_id "
        "FROM placements p JOIN holes h ON h.id=p.hole_id "
        "WHERE p.layout_id=1 AND h.product_id=1 "
        "AND h.x BETWEEN ? AND ? AND h.y BETWEEN ? AND ?",
        (BOARD_X_MIN, BOARD_X_MAX, BOARD_Y_MIN, BOARD_Y_MAX),
    )
    holes = [
        {"id": row[0], "name": row[1], "x": row[2], "y": row[3],
         "set_id": row[4]}
        for row in cur.fetchall()
    ]
    conn.close()
    return holes


def affine_project(holes, params) -> np.ndarray:
    """Project board (x,y) -> pixel (px,py) via a full 2x3 affine matrix M.

    params is either:
      - dict with origin_x/origin_y/scale_x/scale_y/flip_y (axis-aligned seed)
      - 2x3 numpy ndarray (full affine from cv2.estimateAffine2D)
    """
    bx = np.array([h["x"] for h in holes], dtype=np.float64)
    by = np.array([h["y"] for h in holes], dtype=np.float64)
    if isinstance(params, np.ndarray):
        M = params
        out_x = M[0, 0] * bx + M[0, 1] * by + M[0, 2]
        out_y = M[1, 0] * bx + M[1, 1] * by + M[1, 2]
        return np.stack([out_x, out_y], axis=1)
    px = params["origin_x"] + bx * params["scale_x"]
    py = (params["origin_y"] - by * params["scale_y"]) if params["flip_y"] \
        else (params["origin_y"] + by * params["scale_y"])
    return np.stack([px, py], axis=1)


def params_to_dict(M: np.ndarray) -> Dict:
    """Serializable representation of a 2x3 affine matrix."""
    return {
        "affine_matrix": M.tolist(),
        "_note": "Apply as [px,py] = M @ [x,y,1]",
    }


def match_and_refine(
    holes: List[Dict],
    detections: np.ndarray,
    seed: Dict,
    iterations: int = 4,
    tol_px: float = 30.0,
    final_tol_px: float = 10.0,
):
    """Iteratively snap projected holds to nearest detected blob and refit.

    Uses cv2.estimateAffine2D with RANSAC for outlier-robust fitting.
    Returns (final affine matrix, {hole_id: (px, py, radius)}, n_matched).
    """
    params = seed
    detect_xy = detections[:, :2].astype(np.float32)
    radii = detections[:, 2]
    tree = cKDTree(detect_xy)

    matches: Dict[int, Tuple[float, float, float]] = {}
    schedule = np.linspace(tol_px, final_tol_px, iterations)

    for it, cur_tol in enumerate(schedule):
        projected = affine_project(holes, params)
        dists, idxs = tree.query(projected, k=1, distance_upper_bound=float(cur_tol))

        board_pts = []
        pixel_pts = []
        matches.clear()
        for hole, dist, idx in zip(holes, dists, idxs):
            if np.isfinite(dist) and idx < len(detect_xy):
                px, py = detect_xy[idx]
                r = float(radii[idx])
                matches[hole["id"]] = (float(px), float(py), r)
                board_pts.append([hole["x"], hole["y"]])
                pixel_pts.append([px, py])

        if len(board_pts) < 10:
            raise RuntimeError(
                f"Too few matches ({len(board_pts)}) on iteration {it} - "
                "loosen tolerance or check seed calibration."
            )

        src = np.array(board_pts, dtype=np.float32)
        dst = np.array(pixel_pts, dtype=np.float32)
        M, inliers = cv2.estimateAffine2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=4000,
            confidence=0.999,
            refineIters=20,
        )
        if M is None:
            raise RuntimeError("estimateAffine2D failed.")
        n_in = int(inliers.sum()) if inliers is not None else len(src)
        print(f"  iter {it}: tol={cur_tol:.1f}px, matches={len(src)}, "
              f"RANSAC inliers={n_in}")
        params = M

    return params, matches, len(matches)


def interpolate_unmatched(
    holes: List[Dict],
    matches: Dict[int, Tuple[float, float, float]],
    params_main: np.ndarray,
    anchor_set_id: int = 1,
    k: int = 6,
) -> Dict[int, Tuple[float, float, float]]:
    """Fill in pixel coords for unmatched holes by IDW-correcting the
    main affine prediction with the residuals (detected - predicted) of
    the K nearest *matched* holes from the well-calibrated `anchor_set_id`
    (set_id=1 / main T-nut holds, in board space).

    Using only set-1 anchors avoids polluting the interpolation with
    noisy set-20 matches that often snap to the wrong tiny blob.
    """
    anchors = [
        h for h in holes
        if h["set_id"] == anchor_set_id and h["id"] in matches
    ]
    if not anchors:
        return dict(matches)
    anchor_xy = np.array([[h["x"], h["y"]] for h in anchors],
                         dtype=np.float64)
    anchor_px = np.array(
        [[matches[h["id"]][0], matches[h["id"]][1]] for h in anchors],
        dtype=np.float64,
    )
    tree = cKDTree(anchor_xy)
    M = np.asarray(params_main)

    out = dict(matches)
    n_interp = 0
    for h in holes:
        if h["id"] in matches:
            continue
        pred_x = M[0, 0] * h["x"] + M[0, 1] * h["y"] + M[0, 2]
        pred_y = M[1, 0] * h["x"] + M[1, 1] * h["y"] + M[1, 2]
        kk = min(k, len(anchors))
        dists, idxs = tree.query([h["x"], h["y"]], k=kk)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        # Anchor predictions via the main affine.
        nb_xy = anchor_xy[idxs]
        nb_pred_x = M[0, 0] * nb_xy[:, 0] + M[0, 1] * nb_xy[:, 1] + M[0, 2]
        nb_pred_y = M[1, 0] * nb_xy[:, 0] + M[1, 1] * nb_xy[:, 1] + M[1, 2]
        nb_det = anchor_px[idxs]
        res_x = nb_det[:, 0] - nb_pred_x
        res_y = nb_det[:, 1] - nb_pred_y
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
        dx = float((res_x * weights).sum())
        dy = float((res_y * weights).sum())
        out[h["id"]] = (pred_x + dx, pred_y + dy, 0.0)
        n_interp += 1
    print(f"  interpolated {n_interp} unmatched holes from set-{anchor_set_id} anchors")
    return out


def save_debug_overlay(image_bgr: np.ndarray, detections: np.ndarray,
                      holes: List[Dict], matches: Dict, params: Dict,
                      out_path: str) -> None:
    img = image_bgr.copy()
    # All detections in gray.
    for cx, cy, r in detections:
        cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)),
                   (180, 180, 180), 1)
    # Matched holes in green, unmatched DB holds (projected) in red.
    projected = affine_project(holes, params)
    for hole, (epx, epy) in zip(holes, projected):
        if hole["id"] in matches:
            mx, my, mr = matches[hole["id"]]
            cv2.circle(img, (int(round(mx)), int(round(my))),
                       int(round(mr)) + 2, (0, 200, 0), 2)
        else:
            cv2.drawMarker(img, (int(round(epx)), int(round(epy))),
                           (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=1)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=IMAGE_PATH)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--output", default=OUTPUT_JSON)
    ap.add_argument("--debug-image", default=DEBUG_OVERLAY)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--tol", type=float, default=16.0,
                    help="Initial nearest-neighbor tolerance in pixels.")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {args.image}")
    if not os.path.exists(args.db):
        sys.exit(f"DB not found: {args.db}")

    print(f"Loading image: {args.image}")
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        sys.exit("Failed to read image.")
    print(f"  size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    print("Detecting circles…")
    detections = detect_circles(image_bgr)
    print(f"  {len(detections)} candidate circles")

    print("Loading DB holes (layout_id=1)…")
    holes = load_db_holes(args.db)
    print(f"  {len(holes)} holes in board range")
    holes_by_set: Dict[int, List[Dict]] = {}
    for h in holes:
        holes_by_set.setdefault(h["set_id"], []).append(h)
    for sid, hs in holes_by_set.items():
        print(f"    set_id={sid}: {len(hs)} holes")

    # --- Pass 1: main T-nut set (set_id=1) ---
    print("\n[Set 1] Matching + refining affine for main holds…")
    set1_holes = holes_by_set.get(1, [])
    params1, matches1, n1 = match_and_refine(
        set1_holes, detections, SEED,
        iterations=args.iterations, tol_px=args.tol,
    )
    print(f"  matched {n1}/{len(set1_holes)} set-1 holes")
    print(f"  refined params: {params1}")

    # --- Pass 2: screw-on / kickboard set (set_id=20) ---
    # Exclude detections already claimed by set 1 so small foot dots don't get
    # snapped to the (much closer) big-hold screw center.
    used_pts = np.array(
        [[px, py] for (px, py, _r) in matches1.values()],
        dtype=np.float32,
    ) if matches1 else np.zeros((0, 2), dtype=np.float32)
    if len(used_pts):
        used_tree = cKDTree(used_pts)
        d_used, _ = used_tree.query(detections[:, :2], k=1)
        keep_mask = d_used > 6.0  # px
        detections_set2 = detections[keep_mask]
    else:
        detections_set2 = detections
    print(
        f"\n[Set 20] Matching + refining affine for screw-on/kickboard holds…"
    )
    print(f"  candidate detections after excluding set-1 hits: "
          f"{len(detections_set2)}")
    set20_holes = holes_by_set.get(20, [])
    params20, matches20, n20 = match_and_refine(
        set20_holes, detections_set2, params1,
        iterations=max(args.iterations, 5),
        tol_px=40.0, final_tol_px=6.0,
    )
    print(f"  matched {n20}/{len(set20_holes)} set-20 holes")
    print(f"  refined params: {params20}")

    # --- Pass 3: dedicated kickboard (KB1, set_id=20 with y=4) ---
    # KB1 lives on a physically separate strip below the main panel, so
    # its own affine differs from both set-1 and the inter-row set-20
    # screw-ons. Restrict candidate detections to the bottom strip of
    # the image and fit independently.
    kb1_holes = [h for h in set20_holes if h["y"] == BOARD_Y_MIN]
    img_h = image_bgr.shape[0]
    bottom_strip_y = img_h - 200  # px - kickboard sits in last ~200 rows
    bottom_mask = detections[:, 1] >= bottom_strip_y
    if len(used_pts):
        d_used_b, _ = used_tree.query(detections[:, :2], k=1)
        bottom_mask &= (d_used_b > 6.0)
    detections_kb = detections[bottom_mask]
    print(f"\n[KB1] Matching + refining dedicated kickboard affine\u2026")
    print(f"  kickboard holes in DB (y=4): {len(kb1_holes)}")
    print(f"  detections in bottom strip (py>={bottom_strip_y}): "
          f"{len(detections_kb)}")
    params_kb1 = None
    matches_kb1: Dict[int, Tuple[float, float, float]] = {}
    n_kb1 = 0
    if len(kb1_holes) >= 4 and len(detections_kb) >= 4:
        # KB1 holes are all colinear (y=4) so a 2D affine is degenerate.
        # Find the densest horizontal band in the bottom strip — that's
        # the kickboard row — then fit a 1D linear x_board -> px map.
        ys = detections_kb[:, 1]
        # Histogram with 4-px bins to find the kickboard band peak.
        y_lo, y_hi = float(ys.min()), float(ys.max())
        bins = np.arange(y_lo, y_hi + 4.0, 4.0)
        if len(bins) >= 2:
            hist, edges = np.histogram(ys, bins=bins)
            peak = int(hist.argmax())
            band_center = 0.5 * (edges[peak] + edges[peak + 1])
            band_mask = np.abs(ys - band_center) <= 12.0
            band = detections_kb[band_mask]
            print(f"  kickboard band: py\u2248{band_center:.1f} "
                  f"({len(band)} blobs)")

            # Order kickboard holes by board x; order band blobs by px.
            kb_sorted = sorted(kb1_holes, key=lambda h: h["x"])
            band_sorted = band[band[:, 0].argsort()]
            # Initial linear seed: x_board span -> px span of band.
            bx = np.array([h["x"] for h in kb_sorted], dtype=np.float64)
            if len(band_sorted) >= 2 and bx.max() > bx.min():
                a0 = ((band_sorted[:, 0].max() - band_sorted[:, 0].min())
                      / (bx.max() - bx.min()))
            else:
                a0 = float(params1[0, 0])
            b0 = float(band_sorted[:, 0].min() - a0 * bx.min())

            # Iteratively refine: project, snap to nearest band blob,
            # fit np.polyfit on (x_board, px).
            tree_band = cKDTree(band_sorted[:, :2].astype(np.float32))
            cur_a, cur_b = a0, b0
            cur_py = float(np.median(band_sorted[:, 1]))
            for it in range(5):
                proj_px = cur_a * bx + cur_b
                proj_pts = np.column_stack(
                    [proj_px, np.full_like(proj_px, cur_py)]
                ).astype(np.float32)
                tol = max(40.0 - it * 7.0, 8.0)
                d, idx = tree_band.query(proj_pts, k=1,
                                         distance_upper_bound=tol)
                xs_match = []
                pxs_match = []
                pys_match = []
                matches_kb1.clear()
                for h, di, ii in zip(kb_sorted, d, idx):
                    if np.isfinite(di) and ii < len(band_sorted):
                        xs_match.append(h["x"])
                        pxs_match.append(float(band_sorted[ii, 0]))
                        pys_match.append(float(band_sorted[ii, 1]))
                        matches_kb1[h["id"]] = (
                            float(band_sorted[ii, 0]),
                            float(band_sorted[ii, 1]),
                            float(band_sorted[ii, 2]),
                        )
                if len(xs_match) >= 2:
                    cur_a, cur_b = np.polyfit(xs_match, pxs_match, 1)
                    cur_py = float(np.median(pys_match))
                print(f"  iter {it}: tol={tol:.1f}px, "
                      f"matches={len(xs_match)}, "
                      f"a={cur_a:.3f}, b={cur_b:.2f}, py={cur_py:.1f}")

            n_kb1 = len(matches_kb1)
            # Build a 2x3 affine: px depends on x_board only; py constant.
            params_kb1 = np.array(
                [[cur_a, 0.0, cur_b],
                 [0.0, 0.0, cur_py]],
                dtype=np.float64,
            )
            print(f"  matched {n_kb1}/{len(kb1_holes)} KB1 holes")
            print(f"  KB1 1D affine: px = {cur_a:.3f}*x + {cur_b:.2f}, "
                  f"py = {cur_py:.1f}")

    # --- Pass 4: dedicated inter-row screw-ons (set_id=20, y>=20) ---
    # These foot holds sit between main T-nut rows on the same physical
    # panel as set-1, but slightly offset. Fit their own affine using
    # only detections that (a) are not in the kickboard strip and
    # (b) weren't claimed by set-1.
    inter_holes = [h for h in set20_holes if h["y"] >= 20]
    # Use ALL detections above the kickboard strip (no set-1 exclusion).
    # The inter-row screw-on holes sit between set-1 T-nut rows; their
    # DB positions are distinct enough that RANSAC tolerance will keep
    # them from being confused with set-1 T-nut blobs. Excluding set-1
    # hits starves the inter-row pass of valid candidate blobs.
    inter_mask = (detections[:, 1] < bottom_strip_y)
    detections_inter = detections[inter_mask]
    print(f"\n[Set 20 inter-row] Matching + refining dedicated affine\u2026")
    print(f"  inter-row holes (y>=20): {len(inter_holes)}")
    print(f"  candidate detections: {len(detections_inter)}")
    params_inter = None
    matches_inter: Dict[int, Tuple[float, float, float]] = {}
    n_inter = 0
    if len(inter_holes) >= 10 and len(detections_inter) >= 10:
        try:
            params_inter, matches_inter, n_inter = match_and_refine(
                inter_holes, detections_inter, params1,
                iterations=max(args.iterations, 6),
                tol_px=30.0, final_tol_px=5.0,
            )
            print(f"  matched {n_inter}/{len(inter_holes)} inter-row holes")
            print(f"  refined params: {params_inter}")
        except RuntimeError as e:
            print(f"  inter-row fit failed: {e}")
            params_inter = None
            matches_inter = {}

    # Merge results. More specific affines override the generic set-20.
    all_matches: Dict[int, Tuple[float, float, float]] = {}
    all_matches.update(matches1)
    all_matches.update(matches20)
    all_matches.update(matches_inter)
    all_matches.update(matches_kb1)

    # Fill in remaining unmatched holes. KB1 and inter-row holes use
    # their dedicated affines (if available); everything else uses
    # set-1-anchored IDW.
    print("\nInterpolating unmatched holes from set-1 anchors\u2026")
    all_matches = interpolate_unmatched(holes, all_matches, params1,
                                        anchor_set_id=1, k=6)

    if params_inter is not None:
        n_inter_overrides = 0
        for h in inter_holes:
            if h["id"] in matches_inter:
                continue
            pred = affine_project([h], params_inter)[0]
            all_matches[h["id"]] = (float(pred[0]), float(pred[1]), 0.0)
            n_inter_overrides += 1
        print(f"  applied inter-row affine to {n_inter_overrides} "
              f"unmatched inter-row holes")

    if params_kb1 is not None:
        n_kb1_overrides = 0
        for h in kb1_holes:
            if h["id"] in matches_kb1:
                continue  # actually matched, leave alone
            pred = affine_project([h], params_kb1)[0]
            all_matches[h["id"]] = (float(pred[0]), float(pred[1]), 0.0)
            n_kb1_overrides += 1
        print(f"  applied KB1 affine to {n_kb1_overrides} unmatched KB1 holes")

    # Snap interpolated set-20 (foot/screw-on) positions onto the nearest
    # actually-detected small blob, when one exists within tolerance.
    # Inter-row screw-on holes have a small physical offset from the
    # idealized DB coordinate, so this lets us land on the real hold.
    snap_tol = 22.0  # px - tight enough to avoid jumping to a wrong hold
    set1_pts = np.array(
        [[px, py] for hid, (px, py, _r) in matches1.items()],
        dtype=np.float32,
    ) if matches1 else np.zeros((0, 2), dtype=np.float32)
    set1_tree = cKDTree(set1_pts) if len(set1_pts) else None
    all_tree = cKDTree(detections[:, :2].astype(np.float32))
    set20_ids = {h["id"] for h in holes if h["set_id"] == 20}
    kb1_ids = {h["id"] for h in kb1_holes}
    inter_ids = {h["id"] for h in inter_holes}
    snapped = 0
    for hid, (px, py, r) in list(all_matches.items()):
        if hid not in set20_ids or hid in matches20 or hid in matches_kb1 or hid in matches_inter:
            continue
        if hid in kb1_ids:
            continue  # trust KB1 affine, don't snap to a random nearby blob
        if hid in inter_ids and params_inter is not None:
            continue  # trust inter-row affine for the same reason
        d, idx = all_tree.query([px, py], k=5)
        for di, ii in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            if not np.isfinite(di) or di > snap_tol:
                break
            cand_px, cand_py = float(detections[ii, 0]), float(detections[ii, 1])
            cand_r = float(detections[ii, 2])
            if set1_tree is not None:
                ds1, _ = set1_tree.query([cand_px, cand_py], k=1)
                if ds1 < 4.0:  # this blob already belongs to a main hold
                    continue
            all_matches[hid] = (cand_px, cand_py, cand_r)
            snapped += 1
            break
    print(f"  snapped {snapped} foot holes to nearest detected blob")

    hole_set_map = {str(h["id"]): h["set_id"] for h in holes}

    payload = {
        "params": params_to_dict(params1) if isinstance(params1, np.ndarray) else params1,
        "params_by_set": {
            "1": params_to_dict(params1) if isinstance(params1, np.ndarray) else params1,
            "20": params_to_dict(params20) if isinstance(params20, np.ndarray) else params20,
            **({"20_inter": params_to_dict(params_inter)} if params_inter is not None else {}),
            **({"20_kb1": params_to_dict(params_kb1)} if params_kb1 is not None else {}),
        },
        "image": os.path.relpath(args.image, REPO_ROOT),
        "matched_count": len(all_matches),
        "total_holes": len(holes),
        "matched_by_set": {"1": n1, "20": n20, "20_inter": n_inter, "20_kb1": n_kb1},
        "hole_set": hole_set_map,
        "holds": {
            str(hid): {"px": px, "py": py, "r": r}
            for hid, (px, py, r) in all_matches.items()
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {args.output}")

    save_debug_overlay(image_bgr, detections, holes, all_matches, params1,
                       args.debug_image)
    print(f"Debug overlay: {args.debug_image}")


if __name__ == "__main__":
    main()
