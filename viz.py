import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from config import FOOT_CUT_ENABLED, MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING
from kinematics import choose_cut_foot, is_dynamic_hand_move

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


def _draw_hold_scatter(holds: List[Dict], annotate: bool = False, ax=None) -> None:
    """Draw role-colored hold rings on top of the board image (pixel space).

    Each hold uses its detected pixel center and radius (from the calibration
    lookup) when available, falling back to the affine projection otherwise.
    """
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    dpi = fig.dpi if fig is not None else 72.0
    # Ring diameter sized to roughly cover the hold footprint (one grid cell
    # is 4 board units). Detected radii come from screw-dot detection and are
    # too small to use directly.
    base_diameter_px = HOLD_MARKER_SCALE * 4.0 * BOARD_PX_PER_UNIT_X
    # Foot/screw-on holds are physically smaller and sit at half-row offsets
    # between main holds; shrink their ring so it highlights only the foot.
    role_diameter_scale = {15: 0.32}

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


def classify_dynamic_moves(holds: List[Dict], sequence: List[Dict]) -> List[Dict]:
    """For each move index, return a tag describing dynamic/replant status.

    Returns a list of dicts (one per move) with keys:
      - 'is_dynamic': bool — the hand move cut a foot
      - 'cut_limb': 'LF'|'RF'|None — which foot was cut by this move
      - 'is_replant': bool — this foot move re-plants a previously cut foot
    """
    hold_map = _build_hold_map(holds)
    limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
    cut_feet = set()
    tags = []

    for move in sequence:
        limb = move.get('limb')
        hold = hold_map.get(move.get('hold'))
        tag = {'is_dynamic': False, 'cut_limb': None, 'is_replant': False}
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
            if is_dynamic_hand_move(prev_hand_xy, new_xy, foot_xys, limb, other_hand_xy):
                foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                cut = choose_cut_foot(limb, foot_dict)
                if cut is not None:
                    tag['is_dynamic'] = True
                    tag['cut_limb'] = cut
                    cut_feet.add(cut)
                    limb_positions[cut] = None
        elif FOOT_CUT_ENABLED and limb in cut_feet:
            tag['is_replant'] = True
            cut_feet.discard(limb)

        limb_positions[limb] = new_xy
        tags.append(tag)

    return tags


def plot_climb_sequence(
    holds: List[Dict],
    sequence: List[Dict],
    title: str,
    output_path: Optional[str] = None,
    show: bool = False,
    annotate_holds: bool = False,
    include_stats: bool = True,
) -> None:
    if not holds or not sequence:
        raise ValueError("Climb has no holds or sequence")

    _, img_w, img_h = _load_board_image()
    fig_w = 10.0
    fig_h = fig_w * (img_h / img_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _setup_board_axes(ax)

    _draw_hold_scatter(holds, annotate=annotate_holds, ax=ax)

    points_px = _sequence_points_px(holds, sequence)
    dyn_tags = classify_dynamic_moves(holds, sequence)

    # Neutral arrows connecting moves in order. Dynamic moves get a yellow
    # rim and thicker line so the cut/replant pattern is visually obvious.
    arrow_color = "#ffffff"
    arrow_edge = "#000000"
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

    ax.legend(handles=_role_legend_handles(), loc="upper right",
              framealpha=0.85, fontsize=9)
    ax.set_title(title)

    if include_stats:
        start_count = len([h for h in holds if h.get("role_id") == 12])
        hand_count = len([h for h in holds if h.get("role_id") == 13])
        finish_count = len([h for h in holds if h.get("role_id") == 14])
        foot_count = len([h for h in holds if h.get("role_id") == 15])
        stats = compute_sequence_stats(holds, sequence)
        info_lines = [
            f"Moves: {len(sequence)}",
            f"Start Holds: {start_count}",
            f"Hand Holds: {hand_count}",
            f"Finish Holds: {finish_count}",
            f"Foot Holds: {foot_count}",
        ]
        if stats:
            info_lines.extend(
                [
                    f"Avg Move: {stats['avg_distance']:.1f}",
                    f"Max Move: {stats['max_distance']:.1f}",
                    f"Height Gain: {stats['height_gain']:.1f}",
                ]
            )
        dyn_tags = classify_dynamic_moves(holds, sequence)
        n_dyn = sum(1 for t in dyn_tags if t.get('is_dynamic'))
        n_replant = sum(1 for t in dyn_tags if t.get('is_replant'))
        if n_dyn or n_replant:
            info_lines.append(f"Dynamic Moves: {n_dyn}")
            info_lines.append(f"Replants: {n_replant}")
        fig.text(
            0.02,
            0.02,
            "\n".join(info_lines),
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
        )

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
):
    """Interactive step-through viewer overlaid on the Kilter board image."""
    if not holds or not sequence:
        raise ValueError("Climb has no holds or sequence")

    points_px = _sequence_points_px(holds, sequence)
    dyn_tags = classify_dynamic_moves(holds, sequence)

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


def visualizeclimb(climb_data, climb_id, output_dir="climb_visualizations", viz_type="path"):
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
            )
        elif viz_type == "cycle":
            # interactive cycle view (shows a GUI window)
            plot_sequence_cycle(
                holds,
                sequence,
                title=f"{climb.get('name', 'Climb')} (ID: {climb_id})",
                output_path=None,
                show=True,
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
    
    args = parser.parse_args()
    
    try:
        with open(args.data, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    #save
    visualizeclimb(data, args.id, args.output_dir, args.type)

if __name__ == "__main__":
    main()