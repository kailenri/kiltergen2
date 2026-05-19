import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from config import MAX_FOOT_REACH, MAX_HAND_REACH, X_SPACING

ROLE_LABELS = {
    12: "Start",
    13: "Hand",
    14: "Finish",
    15: "Foot",
}

ROLE_COLORS = {
    12: "#2a6fdb",
    13: "#6b6b6b",
    14: "#db3b2a",
    15: "#2a9d8f",
}

LIMB_COLORS = {
    "RH": "#db3b2a",
    "LH": "#2a6fdb",
    "RF": "#2a9d8f",
    "LF": "#6a4c93",
}


def _build_hold_map(holds: List[Dict]) -> Dict[int, Dict]:
    return {h.get("hole_id"): h for h in holds if "hole_id" in h}


def _get_hold_label(hold: Dict) -> str:
    name = hold.get("name")
    if name:
        return str(name)
    hole_id = hold.get("hole_id")
    return str(hole_id) if hole_id is not None else ""


def _draw_hold_scatter(holds: List[Dict], annotate: bool = True) -> None:
    for hold in holds:
        x, y = hold.get("x", 0), hold.get("y", 0)
        role_id = hold.get("role_id", 13)
        color = ROLE_COLORS.get(role_id, "#6b6b6b")
        size = 80 if role_id != 15 else 55
        plt.scatter(x, y, color=color, alpha=0.5, s=size)

        if annotate:
            label = _get_hold_label(hold)
            if label:
                plt.text(x, y - 2, label, fontsize=6, ha="center", alpha=0.7)


def _sequence_points(holds: List[Dict], sequence: List[Dict]) -> List[Tuple[float, float, str]]:
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


def plot_climb_sequence(
    holds: List[Dict],
    sequence: List[Dict],
    title: str,
    output_path: Optional[str] = None,
    show: bool = False,
    annotate_holds: bool = True,
    include_stats: bool = True,
) -> None:
    if not holds or not sequence:
        raise ValueError("Climb has no holds or sequence")

    plt.figure(figsize=(10, 12))
    plt.grid(True, linestyle="--", alpha=0.6)

    _draw_hold_scatter(holds, annotate=annotate_holds)

    points = _sequence_points(holds, sequence)
    for i, (x, y, limb) in enumerate(points):
        plt.scatter(x, y, color=LIMB_COLORS.get(limb, "black"), s=110, zorder=10)
        plt.text(
            x,
            y,
            f"{i + 1}",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    for i in range(1, len(points)):
        prev_x, prev_y, _ = points[i - 1]
        curr_x, curr_y, curr_limb = points[i]
        plt.arrow(
            prev_x,
            prev_y,
            curr_x - prev_x,
            curr_y - prev_y,
            head_width=0.5,
            head_length=0.7,
            fc=LIMB_COLORS.get(curr_limb, "gray"),
            ec=LIMB_COLORS.get(curr_limb, "gray"),
            alpha=0.6,
        )

    limb_labels = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=9, label=limb)
        for limb, color in LIMB_COLORS.items()
    ]
    plt.legend(handles=limb_labels, loc="upper right")

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(title)

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
        plt.figtext(
            0.02,
            0.02,
            "\n".join(info_lines),
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


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
        choices=["path", "reachability-hand", "reachability-foot", "hold-density"],
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