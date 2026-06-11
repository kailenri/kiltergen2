import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import dataclasses

from config import FLAG_ENABLED, FOOT_CUT_ENABLED, MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING
from kinematics import (
    Body,
    choose_cut_foot,
    choose_flag_foot,
    compute_flag_position,
    estimate_hip as _kin_estimate_hip,
    is_crossing_hand_move,
    is_dynamic_hand_move,
    solve_2bone_ik as _kin_solve_2bone_ik,
    _arm_bend_dir as _kin_arm_bend_dir,
    _leg_bend_dir as _kin_leg_bend_dir,
)

# Render-time body proportions. The shared kinematics constants in config.py
# are scaled for the beam-search reachability/ellipse math and over-shoot real
# human limb lengths when used to actually draw a figure. These constants
# render a realistic adult climber (in board-grid units, ~1 unit ≈ 1 inch).
RENDER_TORSO_LEN     = 1.3 * X_SPACING   # ~24 in (hip -> shoulder)
RENDER_HEAD_OFFSET   = 0.55 * X_SPACING  # ~10 in (shoulder -> head center)
RENDER_UPPER_ARM_LEN = 0.7 * X_SPACING   # ~13 in
RENDER_FOREARM_LEN   = 0.65 * X_SPACING  # ~12 in
RENDER_THIGH_LEN     = 1.0 * X_SPACING   # ~19 in
RENDER_SHIN_LEN      = 0.95 * X_SPACING  # ~18 in
RENDER_SHOULDER_HALF = 0.45 * X_SPACING  # ~8 in (half shoulder/hip width)
RENDER_HEAD_RADIUS   = 0.25 * X_SPACING  # ~5 in (head radius in board units)

ROLE_LABELS = {
    12: "Start",
    13: "Hand",
    14: "Finish",
    15: "Foot",
}

# Hold colors by role for board-image overlay.
ROLE_COLORS = {
    12: "#2ecc71",  # Start  -> green
    13: "#3498db",  # Hand   -> blue
    14: "#9b59b6",  # Finish -> purple
    15: "#f1c40f",  # Foot   -> yellow
}

LIMB_COLORS = {
    "RH": "#db3b2a",
    "LH": "#2a6fdb",
    "RF": "#2a9d8f",
    "LF": "#6a4c93",
}

# --- Board-image overlay calibration -----------------------------------------
# Background image of the blank Kilter board. Path is resolved relative to this
# file so it works no matter the CWD.
BOARD_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "base_kilter_image",
    "blank_kilter.jpeg",
)
# Calibration lookup produced by `scripts/calibrate_board_image.py`. Maps each
# DB hole_id to a precise pixel center + radius detected in the board image.
HOLD_PIXEL_MAP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "base_kilter_image",
    "hold_pixel_map.json",
)

# Fallback affine (board -> pixel) used only for holds NOT present in the
# calibration map. Auto-overridden at import time if the map file contains a
# refined affine matrix. Tweakable if you ever regenerate without a map.
BOARD_PX_ORIGIN_X = 0.32
BOARD_PX_ORIGIN_Y = 1376.0
BOARD_PX_PER_UNIT_X = 8.17
BOARD_PX_PER_UNIT_Y = 8.19
BOARD_Y_FLIP = True

# Scales the hold-marker diameter when no per-hold radius is available.
# Used as a multiplier on BOARD_PX_PER_UNIT_X.
HOLD_MARKER_SCALE = 1.8

_BOARD_IMAGE_CACHE: Optional[Tuple] = None
_HOLD_PIXEL_MAP: Optional[Dict] = None  # {hole_id_int: (px, py, r)}
_BOARD_AFFINE: Optional[List[List[float]]] = None  # 2x3 affine from map file
_BOARD_AFFINES_BY_SET: Dict[int, List[List[float]]] = {}  # set_id -> 2x3 affine
_HOLE_SET_MAP: Dict[int, int] = {}  # hole_id -> set_id


def _load_hold_pixel_map() -> Dict[int, Tuple[float, float, float]]:
    """Load the calibration JSON; return {hole_id: (px, py, r)}. Empty dict if missing."""
    global _HOLD_PIXEL_MAP, _BOARD_AFFINE, _BOARD_AFFINES_BY_SET, _HOLE_SET_MAP
    if _HOLD_PIXEL_MAP is not None:
        return _HOLD_PIXEL_MAP

    _HOLD_PIXEL_MAP = {}
    if not os.path.exists(HOLD_PIXEL_MAP_PATH):
        return _HOLD_PIXEL_MAP

    try:
        with open(HOLD_PIXEL_MAP_PATH) as f:
            payload = json.load(f)
    except Exception:
        return _HOLD_PIXEL_MAP

    for hid_str, entry in payload.get("holds", {}).items():
        try:
            _HOLD_PIXEL_MAP[int(hid_str)] = (
                float(entry["px"]),
                float(entry["py"]),
                float(entry.get("r", 0.0)),
            )
        except (TypeError, ValueError, KeyError):
            continue

    params = payload.get("params") or {}
    if isinstance(params, dict) and "affine_matrix" in params:
        _BOARD_AFFINE = params["affine_matrix"]

    # Per-set affines (set_id=1 main holds, set_id=20 screw-on/foot holds).
    by_set = payload.get("params_by_set") or {}
    for sid_str, p in by_set.items():
        if isinstance(p, dict) and "affine_matrix" in p:
            try:
                _BOARD_AFFINES_BY_SET[int(sid_str)] = p["affine_matrix"]
            except (TypeError, ValueError):
                continue

    for hid_str, sid in (payload.get("hole_set") or {}).items():
        try:
            _HOLE_SET_MAP[int(hid_str)] = int(sid)
        except (TypeError, ValueError):
            continue
    return _HOLD_PIXEL_MAP


def _apply_affine(M, x: float, y: float) -> Tuple[float, float]:
    return (M[0][0] * x + M[0][1] * y + M[0][2],
            M[1][0] * x + M[1][1] * y + M[1][2])


def _board_to_px(x: float, y: float, set_id: Optional[int] = None
                 ) -> Tuple[float, float]:
    """Affine fallback: board (x,y) -> image pixel (px,py).

    If `set_id` is given and a per-set affine is loaded, uses it (so
    screw-on/foot holes use their own calibration). Otherwise falls back
    to the main affine, then to the axis-aligned constants.
    """
    _load_hold_pixel_map()
    if set_id is not None and set_id in _BOARD_AFFINES_BY_SET:
        return _apply_affine(_BOARD_AFFINES_BY_SET[set_id], x, y)
    if _BOARD_AFFINE is not None:
        return _apply_affine(_BOARD_AFFINE, x, y)
    px = BOARD_PX_ORIGIN_X + x * BOARD_PX_PER_UNIT_X
    if BOARD_Y_FLIP:
        py = BOARD_PX_ORIGIN_Y - y * BOARD_PX_PER_UNIT_Y
    else:
        py = BOARD_PX_ORIGIN_Y + y * BOARD_PX_PER_UNIT_Y
    return px, py


def _hold_to_px(hold: Dict) -> Tuple[float, float, Optional[float]]:
    """Resolve a climb hold to pixel coords (+ optional detected radius).

    DB-positioning quirk: holes.y is one T-nut row low vs. the official
    Kilter rendering for non-kickboard rows. We compensate by shifting
    y by +8 and routing everything through the set-1 affine (which is
    the most accurate calibration and matches the reference rendering
    for both T-nuts and screw-on feet). Kickboard holes (y < 16) keep
    their per-hole calibrated pixel from the pmap.
    """
    pmap = _load_hold_pixel_map()
    hole_id = hold.get("hole_id")
    x = hold.get("x", 0)
    y = hold.get("y", 0)
    if y < 16:
        if hole_id is not None and hole_id in pmap:
            px, py, r = pmap[hole_id]
            return px, py, (r if r > 0 else None)
        px, py = _board_to_px(x, y, set_id=20)
        return px, py, None
    px, py = _board_to_px(x, y + 8, set_id=1)
    return px, py, None


def _load_board_image():
    global _BOARD_IMAGE_CACHE
    if _BOARD_IMAGE_CACHE is None:
        img = plt.imread(BOARD_IMAGE_PATH)
        height, width = img.shape[0], img.shape[1]
        _BOARD_IMAGE_CACHE = (img, width, height)
    return _BOARD_IMAGE_CACHE


def _setup_board_axes(ax) -> Tuple[int, int]:
    img, width, height = _load_board_image()
    ax.imshow(img, extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image coords: y grows downward
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return width, height


def _role_legend_handles():
    handles = []
    for role_id in (12, 13, 14, 15):
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=ROLE_COLORS[role_id],
                markeredgecolor="black",
                markersize=10,
                label=ROLE_LABELS[role_id],
            )
        )
    return handles


def _build_hold_map(holds: List[Dict]) -> Dict[int, Dict]:
    return {h.get("hole_id"): h for h in holds if "hole_id" in h}


def _get_hold_label(hold: Dict) -> str:
    name = hold.get("name")
    if name:
        return str(name)
    hole_id = hold.get("hole_id")
    return str(hole_id) if hole_id is not None else ""


def _draw_hold_scatter(holds: List[Dict], annotate: bool = False, ax=None,
                       show_direction: bool = True) -> None:
    """Draw role-colored hold rings on top of the board image (pixel space).

    Each hold uses its detected pixel center and radius (from the calibration
    lookup) when available, falling back to the affine projection otherwise.
    When `show_direction` is True and a hold has known orientation, a small
    line is drawn through the hold center aligned with its edge-tangent.
    """
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    dpi = fig.dpi if fig is not None else 72.0
    base_diameter_px = HOLD_MARKER_SCALE * 4.0 * BOARD_PX_PER_UNIT_X
    role_diameter_scale = {15: 0.32}

    orientations = {}
    if show_direction:
        try:
            from sequence_generator import load_hold_orientations
            orientations = load_hold_orientations()
        except Exception:
            orientations = {}

    for hold in holds:
        px, py, _r = _hold_to_px(hold)
        role_id = hold.get("role_id", 13)
        color = ROLE_COLORS.get(role_id, "#6b6b6b")
        diameter_px = base_diameter_px * role_diameter_scale.get(role_id, 1.0)
        diameter_pt = diameter_px * 72.0 / dpi
        marker_area = diameter_pt ** 2

        ax.scatter(
            px, py,
            s=marker_area,
            facecolors="none",
            edgecolors=color,
            linewidths=2.5,
            zorder=4,
        )

        if show_direction and role_id in (12, 13, 14):
            entry = orientations.get(hold.get('hole_id'))
            if entry and float(entry.get('confidence', 0.0)) >= 0.4:
                deg = float(entry.get('direction_deg', 0.0))
                # Edge tangent on image: pixel-y grows DOWNWARD so we negate.
                theta = math.radians(deg)
                r = diameter_px * 0.45
                dx = r * math.cos(theta)
                dy = -r * math.sin(theta)
                ax.plot([px - dx, px + dx], [py - dy, py + dy],
                        color=color, lw=1.4, alpha=0.85, zorder=5)

        if annotate:
            label = _get_hold_label(hold)
            if label:
                ax.text(px, py + diameter_px * 0.6, label,
                        fontsize=6, ha="center", color=color, alpha=0.9)


def _sequence_points(holds: List[Dict], sequence: List[Dict]) -> List[Tuple[float, float, str]]:
    """Return board-coord points (x, y, limb) for each move that maps to a hold."""
    hold_map = _build_hold_map(holds)
    points = []
    for move in sequence:
        hold_id = move.get("hold")
        limb = move.get("limb", "RH")
        if hold_id in hold_map:
            hold = hold_map[hold_id]
            points.append((hold.get("x", 0), hold.get("y", 0), limb))
        elif "x" in move and "y" in move:
            points.append((move.get("x", 0), move.get("y", 0), limb))
    return points


def _sequence_points_px(holds: List[Dict], sequence: List[Dict]) -> List[Tuple[float, float, str]]:
    """Pixel-space sequence points. Uses per-hold lookup when available."""
    hold_map = _build_hold_map(holds)
    out = []
    for move in sequence:
        hold_id = move.get("hold")
        limb = move.get("limb", "RH")
        if hold_id in hold_map:
            px, py, _ = _hold_to_px(hold_map[hold_id])
            out.append((px, py, limb))
        elif "x" in move and "y" in move:
            px, py = _board_to_px(move["x"], move["y"])
            out.append((px, py, limb))
    return out


def compute_sequence_stats(holds: List[Dict], sequence: List[Dict]) -> Dict[str, float]:
    points = _sequence_points(holds, sequence)
    if len(points) < 2:
        return {}

    distances = []
    for i in range(1, len(points)):
        prev_x, prev_y, _ = points[i - 1]
        curr_x, curr_y, _ = points[i]
        distances.append(math.hypot(curr_x - prev_x, curr_y - prev_y))

    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return {
        "avg_distance": sum(distances) / len(distances) if distances else 0,
        "max_distance": max(distances) if distances else 0,
        "height_gain": y_values[-1] - y_values[0],
        "width_span": max(x_values) - min(x_values),
    }


# --- Human-readable beta side panel ----------------------------------------

_ROLE_LABEL_SHORT = {12: "start", 13: "crimp", 14: "FINISH", 15: "foot"}
_LIMB_LABEL_LONG = {"RH": "Right Hand", "LH": "Left Hand",
                    "RF": "Right Foot", "LF": "Left Foot"}


def _direction_phrase(dx: float, dy: float) -> str:
    """Compact phrase describing a move vector. Board units ~ inches."""
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return "match"
    parts = []
    if dy >= 1.0:
        parts.append(f"up {int(round(abs(dy)))}\"")
    elif dy <= -1.0:
        parts.append(f"down {int(round(abs(dy)))}\"")
    if dx >= 1.0:
        parts.append(f"right {int(round(abs(dx)))}\"")
    elif dx <= -1.0:
        parts.append(f"left {int(round(abs(dx)))}\"")
    return " + ".join(parts) if parts else "match"


def _format_beta_steps(
    holds: List[Dict], sequence: List[Dict], dyn_tags: List[Dict]
) -> List[str]:
    """Produce one human-readable beta line per move."""
    hold_map = _build_hold_map(holds)
    limb_positions: Dict[str, Optional[Tuple[float, float]]] = {
        "RH": None, "LH": None, "RF": None, "LF": None,
    }
    lines: List[str] = []
    for i, move in enumerate(sequence):
        limb = move.get("limb", "?")
        limb_name = _LIMB_LABEL_LONG.get(limb, limb)
        hold_id = move.get("hold")
        hold = hold_map.get(hold_id) if hold_id is not None else None
        role = _ROLE_LABEL_SHORT.get(hold.get("role_id"), "hold") if hold else "hold"
        new_xy = (hold.get("x", 0), hold.get("y", 0)) if hold else None

        prev_xy = limb_positions.get(limb)
        if prev_xy is None or new_xy is None:
            phrase = "place on " + role
        else:
            dx = new_xy[0] - prev_xy[0]
            dy = new_xy[1] - prev_xy[1]
            phrase = f"{_direction_phrase(dx, dy)} to {role}"

        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        flags = []
        if tag.get("is_dynamic"):
            cut = tag.get("cut_limb")
            flags.append(f"DYNO, cut {cut}" if cut else "DYNO")
        if tag.get("is_replant"):
            flags.append("replant")
        if tag.get("flag_limb"):
            flags.append(f"flag {tag['flag_limb']}")
        if tag.get("is_unflag"):
            flags.append("unflag")
        suffix = f"  [{', '.join(flags)}]" if flags else ""

        lines.append(f"{i+1:>2}. {limb_name:<11} {phrase}{suffix}")
        if new_xy is not None:
            limb_positions[limb] = new_xy
    return lines


def _build_stats_lines(holds: List[Dict], sequence: List[Dict],
                       dyn_tags: List[Dict]) -> List[str]:
    """Stats block shown below the beta list."""
    start_count = sum(1 for h in holds if h.get("role_id") == 12)
    finish_count = sum(1 for h in holds if h.get("role_id") == 14)
    foot_count = sum(1 for h in holds if h.get("role_id") == 15)
    n_dyn = sum(1 for t in dyn_tags if t.get("is_dynamic"))
    n_replant = sum(1 for t in dyn_tags if t.get("is_replant"))
    stats = compute_sequence_stats(holds, sequence)
    lines = [
        f"Moves: {len(sequence)}",
        f"Holds:  {start_count} start / {finish_count} finish / {foot_count} foot",
    ]
    if stats:
        lines.append(f"Avg move: {stats['avg_distance']:.0f}\"  "
                     f"Max: {stats['max_distance']:.0f}\"")
        lines.append(f"Height gain: {stats['height_gain']:.0f}\"")
    if n_dyn or n_replant:
        lines.append(f"Dynamic: {n_dyn}   Replants: {n_replant}")
    return lines


def _render_beta_panel(
    panel, beta_lines: List[str], stats_lines: List[str],
    active_idx: Optional[int] = None,
) -> None:
    """Draw the beta + stats text into the right-side panel axes.

    `active_idx` highlights one step (used by the GIF). Pass None for the
    static plot to show every step at equal weight.
    """
    panel.clear()
    panel.axis("off")
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)

    panel.text(0.02, 0.985, "BETA", fontsize=12, fontweight="bold",
               va="top", ha="left", color="#222222")

    n = len(beta_lines)
    avail = 0.94 - 0.16
    line_h = min(0.045, avail / max(n, 1))
    font_size = 9 if n <= 14 else (8 if n <= 20 else 7)

    y = 0.94
    for i, line in enumerate(beta_lines):
        if active_idx is None:
            color = "#222222"
            weight = "normal"
            marker = "  "
        elif i == active_idx:
            color = "#000000"
            weight = "bold"
            marker = "\u25B6 "
        elif i < active_idx:
            color = "#888888"
            weight = "normal"
            marker = "  "
        else:
            color = "#bbbbbb"
            weight = "normal"
            marker = "  "
        panel.text(0.02, y, marker + line, fontsize=font_size,
                   family="monospace", va="top", ha="left",
                   color=color, fontweight=weight)
        y -= line_h

    if stats_lines:
        y_stats_top = 0.13
        panel.text(0.02, y_stats_top, "STATS", fontsize=10, fontweight="bold",
                   va="top", ha="left", color="#222222")
        ys = y_stats_top - 0.035
        for line in stats_lines:
            panel.text(0.02, ys, line, fontsize=8, family="monospace",
                       va="top", ha="left", color="#444444")
            ys -= 0.028


def classify_dynamic_moves(holds: List[Dict], sequence: List[Dict]) -> List[Dict]:
    """For each move index, return a tag describing dynamic/replant status.

    Returns a list of dicts (one per move) with keys:
      - 'is_dynamic': bool — the hand move cut a foot
      - 'cut_limb': 'LF'|'RF'|None — which foot was cut by this move
      - 'is_replant': bool — this foot move re-plants a previously cut foot
      - 'flag_limb': 'LF'|'RF'|None — which foot was put into a flag by this move
      - 'is_unflag': bool — this foot move replants from a flag
    """
    hold_map = _build_hold_map(holds)
    limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
    cut_feet = set()
    flags = {'RF': None, 'LF': None}
    tags = []

    for move in sequence:
        limb = move.get('limb')
        hold = hold_map.get(move.get('hold'))
        tag = {'is_dynamic': False, 'cut_limb': None, 'is_replant': False,
               'flag_limb': None, 'is_unflag': False}
        if hold is None or limb not in limb_positions:
            tags.append(tag)
            continue
        new_xy = (hold.get('x', 0), hold.get('y', 0))

        if FOOT_CUT_ENABLED and limb in ('RH', 'LH'):
            prev_hand_xy = limb_positions[limb]
            foot_xys = [limb_positions[f] for f in ('RF', 'LF')
                        if limb_positions[f] is not None]
            other_hand = 'LH' if limb == 'RH' else 'RH'
            other_hand_xy = limb_positions[other_hand]
            cut_happened = False
            if is_dynamic_hand_move(prev_hand_xy, new_xy, foot_xys, limb, other_hand_xy):
                foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                cut = choose_cut_foot(limb, foot_dict)
                if cut is not None:
                    tag['is_dynamic'] = True
                    tag['cut_limb'] = cut
                    cut_feet.add(cut)
                    limb_positions[cut] = None
                    flags[cut] = None
                    cut_happened = True
            if FLAG_ENABLED and not cut_happened:
                if is_crossing_hand_move(new_xy, other_hand_xy, limb):
                    foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                    flag_foot = choose_flag_foot(limb, foot_dict)
                    if flag_foot is not None:
                        planted = [v for v in foot_dict.values() if v is not None]
                        hip = _kin_estimate_hip(planted)
                        if hip is not None:
                            flags[flag_foot] = compute_flag_position(hip, new_xy, flag_foot)
                            limb_positions[flag_foot] = None
                            tag['flag_limb'] = flag_foot
        elif FOOT_CUT_ENABLED and limb in cut_feet:
            tag['is_replant'] = True
            cut_feet.discard(limb)
            flags[limb] = None
        elif limb in ('RF', 'LF') and flags.get(limb) is not None:
            tag['is_unflag'] = True
            flags[limb] = None

        limb_positions[limb] = new_xy
        tags.append(tag)

    return tags


def compute_bodies_along_sequence(
    holds: List[Dict], sequence: List[Dict]
) -> List[Body]:
    """Build a Body snapshot at every move (after applying that move).

    Mirrors classify_dynamic_moves' state tracking so cut feet are excluded
    from hip estimation at the moves where they apply. Also tracks phantom
    flags (off-hold balancing feet) so the body renders the dashed leg.
    """
    hold_map = _build_hold_map(holds)
    limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
    cut_feet = set()
    flags = {'RF': None, 'LF': None}
    bodies: List[Body] = []

    for move in sequence:
        limb = move.get('limb')
        hold = hold_map.get(move.get('hold'))
        if hold is None or limb not in limb_positions:
            bodies.append(Body.from_limb_positions(limb_positions,
                                                   cut_feet=cut_feet,
                                                   flags=flags))
            continue
        new_xy = (hold.get('x', 0), hold.get('y', 0))

        if FOOT_CUT_ENABLED and limb in ('RH', 'LH'):
            prev_hand_xy = limb_positions[limb]
            foot_xys = [limb_positions[f] for f in ('RF', 'LF')
                        if limb_positions[f] is not None]
            other_hand = 'LH' if limb == 'RH' else 'RH'
            other_hand_xy = limb_positions[other_hand]
            cut_happened = False
            if is_dynamic_hand_move(prev_hand_xy, new_xy, foot_xys, limb, other_hand_xy):
                foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                cut = choose_cut_foot(limb, foot_dict)
                if cut is not None:
                    cut_feet.add(cut)
                    limb_positions[cut] = None
                    flags[cut] = None
                    cut_happened = True
            if FLAG_ENABLED and not cut_happened:
                if is_crossing_hand_move(new_xy, other_hand_xy, limb):
                    foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                    flag_foot = choose_flag_foot(limb, foot_dict)
                    if flag_foot is not None:
                        planted = [v for v in foot_dict.values() if v is not None]
                        hip = _kin_estimate_hip(planted)
                        if hip is not None:
                            flags[flag_foot] = compute_flag_position(hip, new_xy, flag_foot)
                            limb_positions[flag_foot] = None
        elif FOOT_CUT_ENABLED and limb in cut_feet:
            cut_feet.discard(limb)
            flags[limb] = None
        elif limb in ('RF', 'LF') and flags.get(limb) is not None:
            flags[limb] = None

        limb_positions[limb] = new_xy
        bodies.append(Body.from_limb_positions(limb_positions,
                                               cut_feet=cut_feet,
                                               flags=flags))

    return bodies


def _body_xy_to_px(xy):
    """Board (x,y) of a body joint -> pixel (px,py).

    Body joints are not real holds; reuse the same y+8 / set_id=1 affine the
    non-kickboard hold path uses so joints align with their parent hand/foot
    markers. Returns None when input is None for plotting convenience.
    """
    if xy is None:
        return None
    return _board_to_px(xy[0], xy[1] + 8, set_id=1)


def _rescale_body_for_render(body: Body) -> Body:
    """Rebuild shoulder/head/elbow/knee using realistic limb proportions.

    Hip and the end-effector positions (rh/lh/rf/lf and their flags) are
    preserved so the figure stays anchored to the active holds.
    """
    if body.hip is None:
        return body
    hip = body.hip
    shoulder = (hip[0], hip[1] + RENDER_TORSO_LEN)
    head = (shoulder[0], shoulder[1] + RENDER_HEAD_OFFSET)
    half = RENDER_SHOULDER_HALF
    r_shoulder = (shoulder[0] + half, shoulder[1])
    l_shoulder = (shoulder[0] - half, shoulder[1])
    r_hip = (hip[0] + half, hip[1])
    l_hip = (hip[0] - half, hip[1])

    r_elbow = None
    l_elbow = None
    if body.rh is not None:
        r_elbow, _ = _kin_solve_2bone_ik(
            r_shoulder, body.rh, RENDER_UPPER_ARM_LEN, RENDER_FOREARM_LEN,
            bend_dir=_kin_arm_bend_dir('RH', r_shoulder, body.rh),
        )
    if body.lh is not None:
        l_elbow, _ = _kin_solve_2bone_ik(
            l_shoulder, body.lh, RENDER_UPPER_ARM_LEN, RENDER_FOREARM_LEN,
            bend_dir=_kin_arm_bend_dir('LH', l_shoulder, body.lh),
        )

    rf_target = body.rf if body.rf is not None else body.rf_flag
    lf_target = body.lf if body.lf is not None else body.lf_flag
    r_knee = None
    l_knee = None
    if rf_target is not None:
        r_knee, _ = _kin_solve_2bone_ik(
            r_hip, rf_target, RENDER_THIGH_LEN, RENDER_SHIN_LEN,
            bend_dir=_kin_leg_bend_dir('RF', r_hip, rf_target),
        )
    if lf_target is not None:
        l_knee, _ = _kin_solve_2bone_ik(
            l_hip, lf_target, RENDER_THIGH_LEN, RENDER_SHIN_LEN,
            bend_dir=_kin_leg_bend_dir('LF', l_hip, lf_target),
        )

    return dataclasses.replace(
        body,
        shoulder=shoulder, head=head,
        r_shoulder=r_shoulder, l_shoulder=l_shoulder,
        r_hip=r_hip, l_hip=l_hip,
        r_elbow=r_elbow, l_elbow=l_elbow,
        r_knee=r_knee, l_knee=l_knee,
    )


def _draw_body(ax, body: Body, alpha: float = 1.0, zorder: int = 9):
    """Render an articulated body on the axes. Returns the list of artists
    so they can be removed before the next frame in the cycle viewer.
    """
    artists = []
    if body.hip is None:
        return artists
    body = _rescale_body_for_render(body)
    if body.shoulder is None:
        return artists

    spine_color = '#222222'
    joint_color = '#111111'

    # Spine: hip -> shoulder -> head.
    spine_pts = [body.hip, body.shoulder]
    if body.head is not None:
        spine_pts.append(body.head)
    spine_px = [_body_xy_to_px(p) for p in spine_pts]
    xs = [p[0] for p in spine_px]
    ys = [p[1] for p in spine_px]
    artists.append(ax.plot(xs, ys, '-', color=spine_color, lw=1.8,
                           alpha=alpha, zorder=zorder)[0])

    # Head: drawn as a filled circle sized in *data* (pixel) coords so it
    # scales correctly with the figure regardless of DPI.
    if body.head is not None:
        hx, hy = _body_xy_to_px(body.head)
        head_radius_px = RENDER_HEAD_RADIUS * BOARD_PX_PER_UNIT_X
        from matplotlib.patches import Circle
        head_circle = Circle(
            (hx, hy), radius=head_radius_px,
            facecolor='white', edgecolor=spine_color,
            linewidth=1.4, alpha=alpha, zorder=zorder + 1,
        )
        ax.add_patch(head_circle)
        artists.append(head_circle)

    # Arms: shoulder -> elbow -> hand.
    for shoulder, elbow, hand, color in [
        (body.r_shoulder, body.r_elbow, body.rh, LIMB_COLORS['RH']),
        (body.l_shoulder, body.l_elbow, body.lh, LIMB_COLORS['LH']),
    ]:
        if shoulder is None or elbow is None or hand is None:
            continue
        pts = [_body_xy_to_px(p) for p in (shoulder, elbow, hand)]
        artists.append(ax.plot([p[0] for p in pts], [p[1] for p in pts],
                               '-', color=color, lw=2.0,
                               alpha=alpha * 0.9, zorder=zorder)[0])
        ex, ey = pts[1]
        artists.append(ax.scatter([ex], [ey], s=18, facecolor=joint_color,
                                  edgecolor='white', linewidth=0.4,
                                  alpha=alpha, zorder=zorder + 1))

    # Legs: hip -> knee -> foot (or flag).
    for hip_pt, knee, foot, flag, color in [
        (body.r_hip, body.r_knee, body.rf, body.rf_flag, LIMB_COLORS['RF']),
        (body.l_hip, body.l_knee, body.lf, body.lf_flag, LIMB_COLORS['LF']),
    ]:
        target = foot if foot is not None else flag
        if hip_pt is None or knee is None or target is None:
            continue
        pts = [_body_xy_to_px(p) for p in (hip_pt, knee, target)]
        line_style = '--' if (foot is None and flag is not None) else '-'
        artists.append(ax.plot([p[0] for p in pts], [p[1] for p in pts],
                               line_style, color=color, lw=2.0,
                               alpha=alpha * 0.9, zorder=zorder)[0])
        kx, ky = pts[1]
        artists.append(ax.scatter([kx], [ky], s=18, facecolor=joint_color,
                                  edgecolor='white', linewidth=0.4,
                                  alpha=alpha, zorder=zorder + 1))
        # Open square at flag endpoint to distinguish from planted foot.
        if foot is None and flag is not None:
            fx, fy = pts[2]
            artists.append(ax.scatter([fx], [fy], s=60, marker='s',
                                      facecolor='none', edgecolor=color,
                                      linewidth=1.4, alpha=alpha,
                                      zorder=zorder + 1))
    return artists


def plot_climb_sequence(
    holds: List[Dict],
    sequence: List[Dict],
    title: str,
    output_path: Optional[str] = None,
    show: bool = False,
    annotate_holds: bool = False,
    include_stats: bool = True,
    draw_body: bool = False,
) -> None:
    if not holds or not sequence:
        raise ValueError("Climb has no holds or sequence")

    _, img_w, img_h = _load_board_image()
    fig_w = 10.0
    fig_h = fig_w * (img_h / img_w)
    # Wider canvas to hold a right-side beta panel beside the board.
    panel_frac = 0.48
    total_w = fig_w * (1.0 + panel_frac)
    fig = plt.figure(figsize=(total_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, panel_frac], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])
    _setup_board_axes(ax)

    _draw_hold_scatter(holds, annotate=annotate_holds, ax=ax)

    points_px = _sequence_points_px(holds, sequence)
    dyn_tags = classify_dynamic_moves(holds, sequence)

    # Neutral arrows connecting moves in order. Dynamic moves get a yellow
    # rim and thicker line so the cut/replant pattern is visually obvious.
    arrow_color = "#ffffff"
    for i in range(1, len(points_px)):
        prev_x, prev_y, _ = points_px[i - 1]
        curr_x, curr_y, _ = points_px[i]
        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        dyn = tag.get('is_dynamic')
        ax.annotate(
            "",
            xy=(curr_x, curr_y),
            xytext=(prev_x, prev_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#f1c40f" if dyn else arrow_color,
                lw=3.2 if dyn else 2.0,
                shrinkA=10, shrinkB=14,
                mutation_scale=14,
            ),
            zorder=8,
        )

    # Numbered step badges on each move.
    for i, (x, y, _limb) in enumerate(points_px):
        ax.scatter(x, y, s=180, facecolor="white", edgecolor="black",
                   linewidth=1.2, zorder=10)
        ax.text(x, y, str(i + 1), fontsize=9, ha="center", va="center",
                color="black", zorder=11, fontweight="bold")

    if draw_body:
        bodies = compute_bodies_along_sequence(holds, sequence)
        if bodies:
            _draw_body(ax, bodies[0], alpha=0.30, zorder=8)
            _draw_body(ax, bodies[-1], alpha=0.95, zorder=9)

    ax.legend(handles=_role_legend_handles(), loc="upper right",
              framealpha=0.85, fontsize=9)
    ax.set_title(title)

    beta_lines = _format_beta_steps(holds, sequence, dyn_tags)
    stats_lines = _build_stats_lines(holds, sequence, dyn_tags) if include_stats else []
    _render_beta_panel(panel, beta_lines, stats_lines, active_idx=None)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_sequence_cycle(
    holds: List[Dict],
    sequence: List[Dict],
    title: str,
    output_path: Optional[str] = None,
    show: bool = True,
    annotate_holds: bool = False,
    draw_body: bool = True,
):
    """Interactive step-through viewer overlaid on the Kilter board image."""
    if not holds or not sequence:
        raise ValueError("Climb has no holds or sequence")

    points_px = _sequence_points_px(holds, sequence)
    dyn_tags = classify_dynamic_moves(holds, sequence)
    bodies = compute_bodies_along_sequence(holds, sequence) if draw_body else []

    _, img_w, img_h = _load_board_image()
    fig_w = 10.0
    fig_h = fig_w * (img_h / img_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.subplots_adjust(bottom=0.12)
    _setup_board_axes(ax)

    _draw_hold_scatter(holds, annotate=annotate_holds, ax=ax)

    # Faded full path (arrows + step badges) underneath the active highlight.
    for i in range(1, len(points_px)):
        px0, py0, _ = points_px[i - 1]
        px1, py1, _ = points_px[i]
        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        dyn = tag.get('is_dynamic')
        ax.annotate(
            "",
            xy=(px1, py1),
            xytext=(px0, py0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#f1c40f" if dyn else "white",
                lw=2.4 if dyn else 1.5,
                alpha=0.55 if dyn else 0.35,
                shrinkA=10, shrinkB=12,
                mutation_scale=12,
            ),
            zorder=6,
        )
    for i, (x, y, _limb) in enumerate(points_px):
        ax.scatter(x, y, s=140, facecolor="white", edgecolor="black",
                   linewidth=1.0, alpha=0.45, zorder=7)
        ax.text(x, y, str(i + 1), fontsize=8, ha="center", va="center",
                color="black", alpha=0.55, zorder=8)

    cur_marker = ax.scatter([], [], s=320, facecolor="white",
                            edgecolor="black", linewidth=1.8, zorder=15)
    cur_label = ax.text(0, 0, "", fontsize=10, ha="center", va="center",
                        color="black", fontweight="bold", zorder=16, visible=False)
    info_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.85),
                        verticalalignment="bottom")

    ax.legend(handles=_role_legend_handles(), loc="upper right",
              framealpha=0.85, fontsize=9)

    idx = {'i': 0}
    body_artists: List = []

    def _clear_body():
        for art in body_artists:
            try:
                art.remove()
            except Exception:
                pass
        body_artists.clear()

    def update():
        i = idx['i']
        x, y, limb = points_px[i]
        cur_marker.set_offsets([[x, y]])
        cur_label.set_position((x, y))
        cur_label.set_text(str(i + 1))
        cur_label.set_visible(True)
        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        suffix = ""
        if tag.get('is_dynamic'):
            suffix = f"  [DYN — cut {tag.get('cut_limb')}]"
        elif tag.get('is_replant'):
            suffix = "  [replant]"
        ax.set_title(f"{title} — Move {i+1}/{len(points_px)} — {limb}{suffix}")
        info_text.set_text(f"Active limb: {limb}\nHold: {sequence[i].get('hold')}{suffix}")
        if draw_body and i < len(bodies):
            _clear_body()
            body_artists.extend(_draw_body(ax, bodies[i], alpha=1.0))
        fig.canvas.draw_idle()

    def prev(event):
        idx['i'] = (idx['i'] - 1) % len(points_px)
        update()

    def nxt(event):
        idx['i'] = (idx['i'] + 1) % len(points_px)
        update()

    axprev = plt.axes([0.3, 0.03, 0.12, 0.05])
    axnext = plt.axes([0.5, 0.03, 0.12, 0.05])
    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')
    bprev.on_clicked(prev)
    bnext.on_clicked(nxt)

    update()

    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_climb_sequence_animated(
    holds: List[Dict],
    sequence: List[Dict],
    title: str,
    output_path: str,
    fps: float = 1.2,
    annotate_holds: bool = False,
    draw_body: bool = False,
    hold_frame: int = 2,
) -> None:
    """Render the climb as an animated GIF that walks through each move with
    a readable beta panel on the right (current step highlighted).

    `hold_frame` repeats the final frame so the viewer sees the completed
    sequence at the end of the loop. Output extension determines writer
    (.gif requires Pillow; .mp4 requires ffmpeg). On writer failure falls
    back to a contact-sheet PNG.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not holds or not sequence:
        raise ValueError("Climb has no holds or sequence")

    points_px = _sequence_points_px(holds, sequence)
    dyn_tags = classify_dynamic_moves(holds, sequence)
    beta_lines = _format_beta_steps(holds, sequence, dyn_tags)
    stats_lines = _build_stats_lines(holds, sequence, dyn_tags)
    bodies = compute_bodies_along_sequence(holds, sequence) if draw_body else []

    _, img_w, img_h = _load_board_image()
    fig_w = 10.0
    fig_h = fig_w * (img_h / img_w)
    panel_frac = 0.48
    total_w = fig_w * (1.0 + panel_frac)
    fig = plt.figure(figsize=(total_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, panel_frac], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])
    _setup_board_axes(ax)
    _draw_hold_scatter(holds, annotate=annotate_holds, ax=ax)

    # Faded full path so the viewer sees the whole route while the active
    # move marker advances through it.
    for i in range(1, len(points_px)):
        px0, py0, _ = points_px[i - 1]
        px1, py1, _ = points_px[i]
        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        dyn = tag.get("is_dynamic")
        ax.annotate(
            "",
            xy=(px1, py1),
            xytext=(px0, py0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#f1c40f" if dyn else "white",
                lw=2.2 if dyn else 1.4,
                alpha=0.55 if dyn else 0.30,
                shrinkA=10, shrinkB=12,
                mutation_scale=12,
            ),
            zorder=6,
        )
    for i, (x, y, _limb) in enumerate(points_px):
        ax.scatter(x, y, s=130, facecolor="white", edgecolor="black",
                   linewidth=1.0, alpha=0.40, zorder=7)
        ax.text(x, y, str(i + 1), fontsize=8, ha="center", va="center",
                color="black", alpha=0.55, zorder=8)

    ax.legend(handles=_role_legend_handles(), loc="upper right",
              framealpha=0.85, fontsize=9)

    cur_marker = ax.scatter([], [], s=320, facecolor="white",
                            edgecolor="black", linewidth=1.8, zorder=15)
    cur_label = ax.text(0, 0, "", fontsize=10, ha="center", va="center",
                        color="black", fontweight="bold", zorder=16, visible=False)

    body_artists: List = []

    def _frame_index(frame: int) -> int:
        return min(frame, len(points_px) - 1)

    def _render(frame: int):
        i = _frame_index(frame)
        x, y, limb = points_px[i]
        cur_marker.set_offsets([[x, y]])
        cur_label.set_position((x, y))
        cur_label.set_text(str(i + 1))
        cur_label.set_visible(True)
        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        suffix = ""
        if tag.get("is_dynamic"):
            suffix = f"  [DYN — cut {tag.get('cut_limb')}]"
        elif tag.get("is_replant"):
            suffix = "  [replant]"
        ax.set_title(f"{title} — Move {i+1}/{len(points_px)} — {limb}{suffix}")
        _render_beta_panel(panel, beta_lines, stats_lines, active_idx=i)
        for art in body_artists:
            try:
                art.remove()
            except Exception:
                pass
        body_artists.clear()
        if draw_body and i < len(bodies):
            body_artists.extend(_draw_body(ax, bodies[i], alpha=1.0))
        return [cur_marker, cur_label, *body_artists]

    n_frames = len(points_px) + max(0, hold_frame)
    interval_ms = max(50, int(1000.0 / max(fps, 0.1)))
    anim = FuncAnimation(
        fig, _render, frames=n_frames, interval=interval_ms, blit=False,
    )

    ext = os.path.splitext(output_path)[1].lower()
    try:
        if ext == ".gif":
            anim.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
        elif ext == ".mp4":
            anim.save(output_path, fps=fps, dpi=110)
        else:
            _save_contact_sheet(fig, points_px, bodies, sequence, dyn_tags,
                                title, output_path)
    except Exception as exc:
        print(f"  [animation] {ext} writer failed ({exc}); writing contact sheet")
        sheet_path = os.path.splitext(output_path)[0] + "_steps.png"
        _save_contact_sheet(fig, points_px, bodies, sequence, dyn_tags,
                            title, sheet_path)

    plt.close(fig)


def _save_contact_sheet(_unused_fig, points_px, bodies, sequence, dyn_tags,
                        title: str, output_path: str) -> None:
    """Fallback when an animation writer is unavailable: render a grid of
    per-step stick-figure frames as a single PNG."""
    n = len(points_px)
    cols = min(4, max(2, n))
    rows = int(math.ceil(n / cols))
    _, img_w, img_h = _load_board_image()
    aspect = img_h / img_w
    fig_w = 4.0 * cols
    fig_h = 4.0 * aspect * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_list = (axes.flat if hasattr(axes, "flat") else [axes]) if n > 1 else [axes]
    for i, ax in enumerate(axes_list):
        if i >= n:
            ax.axis("off")
            continue
        _setup_board_axes(ax)
        # Skip per-hold scatter to keep panels light; just draw the move marker
        # and the stick figure.
        x, y, limb = points_px[i]
        ax.scatter([x], [y], s=180, facecolor="white", edgecolor="black",
                   linewidth=1.2, zorder=10)
        ax.text(x, y, str(i + 1), fontsize=8, ha="center", va="center",
                color="black", fontweight="bold", zorder=11)
        if i < len(bodies):
            _draw_body(ax, bodies[i], alpha=1.0)
        tag = dyn_tags[i] if i < len(dyn_tags) else {}
        suffix = "  [DYN]" if tag.get("is_dynamic") else ""
        ax.set_title(f"{i+1}/{n}  {limb}{suffix}", fontsize=10)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110)
    plt.close(fig)


def _compute_reachability_matrix(holds: List[Dict], mode: str) -> List[List[bool]]:
    coords = [(h.get("x", 0), h.get("y", 0)) for h in holds]
    n = len(holds)
    reachable = [[False for _ in range(n)] for _ in range(n)]
    max_reach = MAX_HAND_REACH if mode == "hand" else MAX_FOOT_REACH

    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if math.hypot(dx, dy) > max_reach:
                continue
            if mode == "hand":
                if holds[i].get("role_id") not in {12, 13, 14}:
                    continue
                if holds[j].get("role_id") not in {12, 13, 14}:
                    continue
            else:
                # Avoid large downward steps for feet, matching generator behavior.
                if holds[j].get("role_id") != 12:
                    if holds[j].get("y", 0) < holds[i].get("y", 0) - X_SPACING * 1.5:
                        continue
            reachable[i][j] = True

    return reachable


def plot_reachability_map(
    holds: List[Dict],
    mode: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    if not holds:
        raise ValueError("No holds to visualize")

    mode = "hand" if mode == "hand" else "foot"
    matrix = _compute_reachability_matrix(holds, mode)
    counts = [sum(1 for v in row if v) for row in matrix]

    x_vals = [h.get("x", 0) for h in holds]
    y_vals = [h.get("y", 0) for h in holds]

    plt.figure(figsize=(10, 12))
    plt.grid(True, linestyle="--", alpha=0.4)

    scatter = plt.scatter(
        x_vals,
        y_vals,
        c=counts,
        cmap="viridis",
        s=90,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    plt.colorbar(scatter, label="Reachable Holds")

    for hold in holds:
        label = _get_hold_label(hold)
        if label:
            plt.text(hold.get("x", 0), hold.get("y", 0) - 2, label, fontsize=6, ha="center")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    if not title:
        title = f"{mode.capitalize()} Reachability Map"
    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def plot_hold_density(
    holds: List[Dict],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    if not holds:
        raise ValueError("No holds to visualize")

    x_vals = [h.get("x", 0) for h in holds]
    y_vals = [h.get("y", 0) for h in holds]

    plt.figure(figsize=(10, 12))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.hist2d(x_vals, y_vals, bins=20, cmap="magma")
    plt.colorbar(label="Hold Density")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(title or "Hold Density Heatmap")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def visualizeclimb(climb_data, climb_id, output_dir="climb_visualizations", viz_type="path", draw_body=True):
    os.makedirs(output_dir, exist_ok=True)

    climb = None
    for result in climb_data.get("results", []):
        if result.get("id") == climb_id:
            climb = result
            break

    if not climb:
        print(f"Climb with ID '{climb_id}' not found")
        return False

    climb_name = climb.get("name", "Unnamed").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = viz_type.replace("-", "_")
    save_path = os.path.join(output_dir, f"{climb_name}_{suffix}_{timestamp}.png")

    print(f"Visualizing climb: {climb.get('name', 'Unnamed')} (ID: {climb_id})")

    holds = climb.get("best_sequence", {}).get("holds", [])
    sequence = climb.get("best_sequence", {}).get("sequence", [])

    try:
        if viz_type == "path":
            plot_climb_sequence(
                holds,
                sequence,
                title=f"{climb.get('name', 'Climb')} (ID: {climb_id})",
                output_path=save_path,
                draw_body=draw_body,
            )
        elif viz_type == "cycle":
            # interactive cycle view (shows a GUI window)
            plot_sequence_cycle(
                holds,
                sequence,
                title=f"{climb.get('name', 'Climb')} (ID: {climb_id})",
                output_path=None,
                show=True,
                draw_body=draw_body,
            )
        elif viz_type == "reachability-hand":
            plot_reachability_map(
                holds,
                mode="hand",
                output_path=save_path,
                title=f"Hand Reachability - {climb.get('name', 'Climb')}",
            )
        elif viz_type == "reachability-foot":
            plot_reachability_map(
                holds,
                mode="foot",
                output_path=save_path,
                title=f"Foot Reachability - {climb.get('name', 'Climb')}",
            )
        elif viz_type == "hold-density":
            plot_hold_density(
                holds,
                output_path=save_path,
                title=f"Hold Density - {climb.get('name', 'Climb')}",
            )
        else:
            print(f"Unknown viz type: {viz_type}")
            return False
    except ValueError as exc:
        print(str(exc))
        return False

    print(f"Visualization saved to {save_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize a specific climb and save the image")
    parser.add_argument("--data", required=True, help="Path to the climbing data JSON file")
    parser.add_argument("--id", required=True, help="ID of the climb to visualize")
    parser.add_argument("--output-dir", default="climb_visualizations", 
                       help="Directory to save visualizations")
    parser.add_argument(
        "--type",
        default="path",
        choices=["path", "cycle", "reachability-hand", "reachability-foot", "hold-density"],
        help="Type of visualization to generate",
    )
    parser.add_argument(
        "--body", dest="draw_body", action="store_true", default=True,
        help="Overlay articulated body skeleton (default).",
    )
    parser.add_argument(
        "--no-body", dest="draw_body", action="store_false",
        help="Disable the body overlay.",
    )

    args = parser.parse_args()
    
    try:
        with open(args.data, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    #save
    visualizeclimb(data, args.id, args.output_dir, args.type, draw_body=args.draw_body)

if __name__ == "__main__":
    main()